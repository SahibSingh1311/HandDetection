import handy
import cv2
cap = cv2.VideoCapture(0)

# capture the hand histogram by placing your hand in the box shown and
# press 'A' to confirm
hist = handy.capture_histogram(source=0)

while True:
    ret, frame = cap.read()
    if not ret:
        
        break

    handy.detect_face(frame, block=False)

    # detect the hand
    hand = handy.detect_hand(frame, hist)

    # to get the outline of the hand
    custom_outline = hand.draw_outline(
        min_area=10000, color=(0, 255, 255), thickness=2)

    # to get a quick outline of the hand
    quick_outline = hand.outline

    # draw fingertips on the outline of the hand, with radius 5 and color red,
    # filled in.
    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)

    # to get the centre of mass of the hand
    com = hand.get_center_of_mass()
    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)

    cv2.imshow("Handy", quick_outline)

    # display the unprocessed, segmented hand
    # cv2.imshow("Handy", hand.masked)

    # display the binary version of the hand
    # cv2.imshow("Handy", hand.binary)

    k = cv2.waitKey(5)

    # Press 'q' to exit
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
