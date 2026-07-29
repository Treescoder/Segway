class PathFollowerBase:
    def update(self, pos, heading, dt):
        raise NotImplementedError

    def plot_desired(self, ax, traj_x=None):
        pass