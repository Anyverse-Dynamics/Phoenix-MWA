"""Common StreamSample metadata keys.

这些字段只描述数据事实本身，不能表达业务状态或控制命令。
"""

META_KIND = "kind"
META_STREAM_NAME = "stream_name"
META_SENSOR = "sensor"
META_FIELD = "field"
META_PART = "part"
META_LOSSLESS = "lossless"
META_STAMP_SOURCE = "stamp_source"

KIND_ROBOT_STATE = "robot_state"
KIND_ROBOT_EEF = "robot_eef"
KIND_SENSOR_IMAGE = "sensor_image"
KIND_SENSOR_INFO = "sensor_info"
KIND_SENSOR_POINTCLOUD = "sensor_pointcloud"
KIND_INPUT = "input"

STAMP_SOURCE_ROS_HEADER = "ros_header"
STAMP_SOURCE_RECV_TIME = "recv_time"
STAMP_SOURCE_ROBOT_READ_DONE = "robot_read_done"
