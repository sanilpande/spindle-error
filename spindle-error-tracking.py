import cv2
import sys
import matplotlib.pyplot as plt
import time
import csv
from scipy import optimize
import numpy as np
import math



if __name__ == '__main__':
 
    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[5]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture("/home/sanil/Desktop/mchnvid/250m.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 300, param1=500, param2=50, minRadius=0, maxRadius=500)
    print(circles)

    # Uncomment the line below to select a different bounding box
    #    bbox = cv2.selectROI(frame, True)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        #    	 show the output image
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1366, 768)
        cv2.imshow("image", frame)
        cv2.imwrite('detected.png',frame)
        cv2.waitKey(0)
    #    print(circles[0,0]-30, circles[0,1]-30, circles[0,0]+30,circles[0,1]+30)
    # Define an initial bounding box
    bbox = (circles[0, 0] - 200, circles[0, 1] - 200, 400, 400)
    #    print(bbox)

    # conversion in mm
    pix2mm = 46 / (circles[0, 2] * 2)
    print(pix2mm)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    xs = []
    ys = []
    t = []
    time_inital = time.time()
    radius = []

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        x_temp = (bbox[0] + bbox[2] / 2) * pix2mm
        y_temp = (bbox[1] + bbox[3] / 2) * pix2mm
        xs.append(x_temp)
        ys.append(y_temp)

        timex = time.time() - time_inital
        t.append(timex)
        #        data=[p1[0],p1[1],timex]
        #        full_data.append(data)

        time.sleep(1 / 50)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or timex >= 40:
            video.release()
            break
    cv2.destroyAllWindows()
    x_mean = sum(xs) / len(xs)
    print(x_mean)
    #    print(len/t(len))
    y_mean = sum(ys) / len(ys)

    y_amp = []
    x_amp = []
    for a in ys:
        y_amp.append(a - y_mean)
    #    print(y_amp)

    for a in xs:
        x_amp.append(a - x_mean)
    #    print(x_amp)
    for a in range(0, len(x_amp)):
        xx = x_amp[a]
        yy = y_amp[a]
        radius.append(math.sqrt(xx ** 2 + yy ** 2))
    #    print (radius)
    mean_deviation = sum(radius) / len(radius)

    plt.figure(0)

    plt.subplot(211)
    plt.plot(t, x_amp, color='green')
    plt.xlim(0, 40)
    plt.ylim(-3, 3)
    plt.ylabel('Deviation of X (mm)')
    plt.title('Deviation vs. Time')

    plt.subplot(212)
    plt.plot(t, y_amp, color='blue')
    plt.xlim(0, 40)
    plt.ylim(-3, 3)
    plt.xlabel('Time (s)')
    plt.ylabel('Deviation of Y (mm)')

    #    plt.savefig("/home/sanil/Desktop/GRAPHS/mosse/mosse_spindle.png")

    plt.figure(1)
    plt.plot(t, radius, 'k.')
    plt.axhline(y=mean_deviation, color='r', linestyle='-')
    plt.ylim(0, 3)
    plt.xlim(0, 40)
    plt.xlabel('Time (s)')
    plt.text(3, 0.5, "Mean Radial Error =" + str(round(mean_deviation, 3)) + "mm", family="sans-serif")
    plt.ylabel('Radial Deviation (mm)')
    plt.title('Radial Deviation vs. Time')

    #    plt.savefig("/home/sanil/Desktop/GRAPHS/mosse/mosse_spindle_radius.png")

    plt.figure(2)

    plt.subplot(211)
    plt.plot(t, x_amp, '.', color='green')
    plt.xlim(0, 40)
    plt.ylim(-3, 3)
    plt.ylabel('Deviation of X (mm)')
    plt.title('Deviation vs. Time')

    plt.subplot(212)
    plt.plot(t, y_amp, '.', color='blue')
    plt.xlim(0, 40)
    plt.ylim(-3, 3)
    plt.xlabel('Time (s)')
    plt.ylabel('Deviation of Y (mm)')
    #    plt.savefig("/home/sanil/Desktop/GRAPHS/mosse/mosse_spindle_scatter.png")

    plt.show()

    table = [t, x_amp, y_amp]
    with open('/home/sanil/Desktop/Table/mosse_spindle_fake.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(table)
    csvFile.close()

