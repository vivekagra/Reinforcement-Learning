# define action space and command it to robot

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_msgs.msg import Int16

class Cmd_action_to_robot():
    def __init__(self):
        self.init_node('action')
        
        self.action_sub      = rospy.Subscriber('cmd_action', Int16, take_action_cb)
        
        self.cmd_vel_pub     = rospy.Pubscriber('cmd_vel', Twist, queue_size=1)
        #self.linear_vel_pub  = rospy.Publisher('linear_vel', Float32, queue_size=10)
        #self.angular_vel_pub = rospy.Publisher('angular_vel', Float32, queue_size=10)
        
        self.linear          = 0
        self.angular         = 0
            

    def take_action_cb(self,msg):
        action = msg.data
        if action == 0:
            self.linear  = 0
            self.angular = 0
        elif action == 1:
            self.linear  = 1.5
            self.angular = 1
        elif action == 2:
            self.linear  = 2
            self.angular = 0.75
        elif action == 3:
            self.linear  = 2
            self.angular = 0
        elif action == 4:
            self.linear  = 2
            self.angular = -0.75
        else:
            self.linear  = 1.5
            self.angular = -1
            
    def update(self):
        move_cmd = Twist()
        move_cmd.linear.x = self.linear
        move_cmd.angular.z = self.angular
        self.cmd_vel_pub.publish(move_cmd)
        
    def shutdown(self):
        rospy.loginfo("Action Space Closed")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
        
    def spin(self):
        rospy.loginfo("Action Space Initialized")
        rate = rospy.Rate(10)
        rospy.on_shutdown(self.shutdown)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()
        rospy.spin()
        
def main():
    action_spc = Cmd_action_to_robot()
    action_spc.spin()
    
if __name__ == '__main__':
    main()