#include "data_names.h"

namespace Sensors {
const char* names[] = {
  "body_orientation",
  "body_accel",
  "body_gyro",
  "wheel1_ground_force",
  "wheel2_ground_force",
  "wheel_joint_angle1",
  "wheel_joint_angle2",
  "wheel_joint_speed1",
  "wheel_joint_speed2",
};

const int count = sizeof(names) / sizeof(names[0]);
}  // namespace Sensors

namespace Joints {
const char* names[] = {
  "alphaL",
  "betaL",
  "wheel1",
  "alphaR",
  "betaR",
  "wheel2",
};

const int count = sizeof(names) / sizeof(names[0]);
}  // namespace Joints

namespace Actuators {
const char* names[] = {
  "motor_alphaL",
  "motor_betaL",
  "motor_alphaR",
  "motor_betaR",
  "motor_wheelL",
  "motor_wheelR",
};

const int count = sizeof(names) / sizeof(names[0]);
}  // namespace Actuators
