import mujoco.viewer

def main():
    model = mujoco.MjModel.from_xml_path('xml/scene.xml')
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()