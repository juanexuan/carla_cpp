# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
这个模块实现了一个智能体（agent），它能在赛道上随机沿着路径点漫游，同时避开其他车辆，并且能对交通信号灯做出响应。
该智能体还可以利用全局路径规划器来遵循指定的路线。
"""

import carla
# 从agents.navigation.basic_agent模块中导入BasicAgent类，后续会基于它进行拓展实现新的智能体类
from agents.navigation.basic_agent import BasicAgent
# ConstantVelocityAgent类继承自BasicAgent类，用于实现以固定速度在场景中导航的智能体
class ConstantVelocityAgent(BasicAgent):
    """
    ConstantVelocityAgent类实现了一个以固定速度在场景中导航的智能体。
    如果要求该智能体以期望速度执行不可能的转弯（包括变道）操作，它将会失败。
    当检测到碰撞时，固定速度行驶行为会停止，等待一段时间后再重新启动。
    """

    def __init__(self, vehicle, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None):
        """
        初始化智能体的参数、本地规划器和全局规划器。

        参数说明：
        :param vehicle: 要应用智能体逻辑的车辆角色（actor），也就是智能体要控制的车辆对象。
        :param target_speed: 车辆行驶的目标速度（单位为千米/小时），用于指定智能体期望车辆运行的速度。
        :param opt_dict: 字典类型，用于在需要更改某些参数时进行设置，这也适用于与本地规划器相关的参数。
        :param map_inst: carla.Map实例，传入此实例可以避免重复获取地图对象的开销较大的调用。
        :param grp_inst: GlobalRoutePlanner实例，传入此实例可避免获取全局路径规划器对象时开销较大的调用。
        """
        # 调用父类（BasicAgent）的构造函数来初始化一些基础属性和功能，传递相关参数
        super().__init__(vehicle, target_speed, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        # 是否在固定速度行驶行为停止时使用BasicAgent的行为，初始化为False
        self._use_basic_behavior = False  # Whether or not to use the BasicAgent behavior when the constant velocity is down
         # 将目标速度从千米/小时转换为米/秒，方便后续计算，存储在实例变量中
        self._target_speed = target_speed / 3.6  # [m/s]
         # 获取当前车辆的速度长度（大小），单位为米/秒，作为当前速度初始值
        self._current_speed = vehicle.get_velocity().length()  # [m/s]
         # 用于记录固定速度行为停止的时间，初始化为None
        self._constant_velocity_stop_time = None
         # 碰撞传感器对象，初始化为None，后续会进行创建和设置
        self._collision_sensor = None
         # 碰撞后再次启动固定速度行为之前等待的时间，初始化为正无穷大，后续可根据配置进行修改

        self._restart_time = float('inf')  # 如果在opt_dict字典中存在'restart_time'键，说明用户传入了自定义的重启时间，更新实例变量的值

        if 'restart_time' in opt_dict:
            self._restart_time = opt_dict['restart_time']
            # 如果在opt_dict字典中存在'use_basic_behavior'键，说明用户传入了是否使用基本行为的配置，更新相应实例变量的值
        if 'use_basic_behavior' in opt_dict:
            self._use_basic_behavior = opt_dict['use_basic_behavior']
            # 标记固定速度行为是否处于激活状态，初始化为True

        self.is_constant_velocity_active = True
         # 设置碰撞传感器，用于检测车辆是否发生碰撞
        self._set_collision_sensor()
        # 设置固定速度，使车辆按照目标速度行驶
        self._set_constant_velocity(target_speed)

    def set_target_speed(self, speed):
        """改变智能体的目标速度（单位为千米/小时）。

        参数：
        :param speed: 新的目标速度值（千米/小时），用于更新智能体期望车辆达到的速度。"""
         # 将传入的千米/小时速度转换为米/秒，并更新目标速度实例变量
        self._target_speed = speed / 3.6
         # 通过本地规划器设置速度，使其生效
        self._local_planner.set_speed(speed)

    def stop_constant_velocity(self):
        """停止固定速度行驶行为。"""
        # 将固定速度行为的激活状态标记为False，表示停止
        self.is_constant_velocity_active = False
        # 调用车辆对象的方法，禁用固定速度行驶功能
        self._vehicle.disable_constant_velocity()
         # 获取当前世界的时间快照，记录固定速度行为停止的时间（以秒为单位）
        self._constant_velocity_stop_time = self._world.get_snapshot().timestamp.elapsed_seconds

    def restart_constant_velocity(self):
        """Public method to restart the constant velocity"""
        self.is_constant_velocity_active = True
        self._set_constant_velocity(self._target_speed)

    def _set_constant_velocity(self, speed):
        """Forces the agent to drive at the specified speed"""
        self._vehicle.enable_constant_velocity(carla.Vector3D(speed, 0, 0))

    def run_step(self):
        """Execute one step of navigation."""
        if not self.is_constant_velocity_active:
            if self._world.get_snapshot().timestamp.elapsed_seconds - self._constant_velocity_stop_time > self._restart_time:
                self.restart_constant_velocity()
                self.is_constant_velocity_active = True
            elif self._use_basic_behavior:
                return super(ConstantVelocityAgent, self).run_step()
            else:
                return carla.VehicleControl()

        hazard_detected = False

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_speed = self._vehicle.get_velocity().length()

        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, adversary, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            vehicle_velocity = self._vehicle.get_velocity()
            if vehicle_velocity.length() == 0:
                hazard_speed = 0
            else:
                hazard_speed = vehicle_velocity.dot(adversary.get_velocity()) / vehicle_velocity.length()
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + 0.3 * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_speed = 0
            hazard_detected = True

        # The longitudinal PID is overwritten by the constant velocity but it is
        # still useful to apply it so that the vehicle isn't moving with static wheels
        control = self._local_planner.run_step()
        if hazard_detected:
            self._set_constant_velocity(hazard_speed)
        else:
            self._set_constant_velocity(self._target_speed)

        return control

    def _set_collision_sensor(self):
        blueprint = self._world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(blueprint, carla.Transform(), attach_to=self._vehicle)
        self._collision_sensor.listen(lambda event: self.stop_constant_velocity())

    def destroy_sensor(self):
        if self._collision_sensor:
            self._collision_sensor.destroy()
            self._collision_sensor = None
