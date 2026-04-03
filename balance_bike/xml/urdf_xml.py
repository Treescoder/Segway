import mujoco

model = mujoco.MjModel.from_xml_path("D:\Work_Learn\Segway\\balance_bike\\xml\\balance_bike.urdf")

mujoco.mj_saveLastXML("D:\Work_Learn\Segway\\balance_bike\\xml\\balance_bike.xml", model)