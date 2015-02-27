from Tkinter import Tk, Canvas, PhotoImage, mainloop

from matrix import *

def showxy(event):
	global img

	xm, ym = event.x, event.y
	str1 = "mouse at x=%d  y=%d" % (xm, ym)
	# Show coordinates in title
	root.title(str1)
	# Switch color to red if mouse enters a set location range
	x, y, delta = 100, 100, 10
	#frame.config(bg='red' if abs(xm - x) < delta and abs(ym - y) < delta else 'yellow')

	(xest, yest) = kalman_filter([xm, ym])
	
	#print "predicted values x=%d  y=%d" % (xest, yest)

	img.put("#FF0000", (int(round(xest)), int(round(yest))))


def kalman_filter(measurement):
	global x, P

	# prediction
	x = (F * x) + u
	P = F * P * F.transpose() + Q

	# measurement update
	Z = matrix([measurement])
	y = Z.transpose() - (H * x)
	S = H * P * H.transpose() + R
	K = P * H.transpose() * S.inverse()
	x = x + (K * y)
	P = (I - (K * H)) * P
    
	#print "\n"
	#print x

	return (x.value[0][0], x.value[1][0])
	#return (P.value[0][0], P.value[1][1])

dt = 0.1

Q_angle = abs(random.random()*0.)
Q_gyroBias = abs(random.random()*0.)
Q3 = abs(random.random()*10000)
Q4 = abs(random.random()*100)

x = matrix([[0.], [0.], [0.], [0.]]) # initial state (location and velocity)
u = matrix([[0.], [0.], [0.], [0.]]) # external motion

P =  matrix([[1000., 0., 0., 0.], [0., 1000., 0., 0.], [0., 0., 1000., 0.], [0., 0., 0., 1000.]])# initial uncertainty
F =  matrix([[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0., 0., 0., 1.]]) # next state function
H =  matrix([[1., 0., 0., 0.], [0., 1., 0., 0.]])# measurement function

ra = 10.0**2
R =  matrix([[ra, 0.], [0., ra]]) # measurement uncertainty

I =  matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]) # identity matrix

sv = 8.8

G = matrix([[0.5*dt**2],
	        [0.5*dt**2],
	        [dt],
	        [dt]])

#Q = G*G.transpose()
Q =  matrix([[Q_angle, 0., 0., 0.], [0., Q_gyroBias, 0., 0.], [0., 0., Q3, 0.], [0., 0., 0., Q4]])

root = Tk()

"""
frame = tk.Frame(root, bg='yellow', width=300, height=200)
frame.bind("<Motion>", showxy)
frame.pack()
"""

WIDTH = 640
HEIGHT = 480

canvas = Canvas(root, width=WIDTH, height=HEIGHT, bg="#000000")
canvas.bind("<Motion>", showxy)
canvas.pack()
img = PhotoImage(width=WIDTH, height=HEIGHT)
canvas.create_image((WIDTH/2, HEIGHT/2), image=img, state="normal")

root.mainloop()