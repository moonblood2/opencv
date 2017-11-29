import cv2
import numpy as np
import os

os.chdir("C:\\Users\\ASUS\\Desktop\\opencv")
img = cv2.imread('img.jpg',cv2.IMREAD_COLOR)
gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##pts = np.array([[0+10,200],[200,400-10],[400-10,200],[200,0+10]],np.int32)
##pts = pts.reshape((-1,1,2))
##cv2.polylines(gimg,[pts], True, (0,255,0), 1)
roi = img[10:390,10:390]
cv2.imshow('gimg',roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
##print(img.shape)
##gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##cv2.imshow('img',gimg)
##px = img[55,55]
##
##print(px)

for the past this year i've been working with my dad -a fammily bussiness-
he is a contractor -a contruction company- not that big around 75 employee
so basicly I've to go around the construction site to monitor the worker
and make sure that they are doing the right thing
and also doing almost all the paperwork like calculate the employee salary
or making an Invoice(receipt) and tax for the client
and doing the worik documentation for the forienger, like because they are
a foreigner so they can't stay in thailand over 90 days
so I have renew theur visa that's it

I know this may seem like it's not related to IT feild at all right? but it is
because I have an access to all of the documents and have to do all the paperwork
so I have idea to make it easier
Frist of all I ceated the database which contain the employee information like
name, salary rate, how many ours they've worked Ot hours.
and I made a webpage for my father so that may be it will be easier for him
to look at this data or modify the data
but when I show him the website and teach how to use, like you need enter the password
and use this form to access or modify the data
and he was like this is so unnessessery and too complicated I can remember everything
about my employee, won't you atleast try it, No
and I start to relize maybe i'm worng, maybe my program is better than
Using the Microsoft excel like always (20 years) is the right thing

But I'm not gonna give up, I've move to the next project the machine learning
why? you need to know about my father's employee, you know some of them
maybe not very honest about the amount of work they or OT hours which wil be
used to calculate into their salary. so they just added 1 or 2 hours
without my father noticed. So I'm using the classifier in machinelearning
like knearest neigbor, K means to make a cluster of the employees
what this means is the empolyee who cheat about their work will not fall into any of this group
you know they will be outstanding
But by all means this doesn't make him a cheater. It just make it easier for me to keep an eyes on the right person
and other problem than that will be like, ahh every employee can take some of their future salary before the pay day.
but there can be someone who takes that money and just be gone, and we never see them again.
I've try to predict wheather this will happened or not using the machine learning.
But it's not success.We doesn't have enough data
'''











