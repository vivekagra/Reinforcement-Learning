# this function process information from environment and give reward to our bot
# environment information include camera input, depth information, distance from target object.

import rospy
from std_msgs.msg import Float32

class give_reward_to_robot():
    def __init__(self):
        rospy.init_node('reward')
        
        rospy.Publisher('reward_robot', Float32, queue_size=1)

    def shutdown(self):
        rospy.loginfo("Reward Function Ended")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
        
    def spin(self):
        rospy.loginfo("Reward Space Initialized")
        rate = rospy.Rate(10)
        rospy.on_shutdown(self.shutdown)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()
        rospy.spin()
        
def main():
    reward_robot = give_reward_to_robot()
    reward_robot.spin()
if __name__ == '__main__':
    main()