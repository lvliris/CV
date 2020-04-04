import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
import sys
from queue import Queue


def get_angle(x, y):
    r = np.sqrt(x**2 + y**2)
    angle = np.arcsin(y / r)
    if x > 0 and y >= 0:
        return angle
    elif x <= 0 and y > 0:
        return np.pi - angle
    elif x < 0 and y <= 0:
        return np.pi - angle
    else:
        return 2*np.pi + angle


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scale):
        return Vector(scale*self.x, scale*self.y)

    def __rmul__(self, scale):
        return self * scale

    def __truediv__(self, scale):
        return Vector(self.x / scale, self.y / scale)

    def __str__(self):
        return '({:4f}, {:4f})'.format(self.x, self.y)

    def get_l2norm(self):
        return self.x**2 + self.y**2

    def get_unit_vector(self):
        module = np.sqrt(self.get_l2norm())
        return self / module

    def get_angle(self):
        x = self.x
        y = self.y
        r = np.sqrt(x ** 2 + y ** 2)
        angle = np.arcsin(y / r)
        if x > 0 and y >= 0:
            return angle
        elif x <= 0 and y > 0:
            return np.pi - angle
        elif x < 0 and y <= 0:
            return np.pi - angle
        else:
            return 2 * np.pi + angle

class Star(object):
    G = 1e-3
    index = 0
    def __init__(self, mass, position: Vector, velocity: Vector, name=None, color='r'):
        self.m = mass
        self.pos = position
        self.v = velocity
        if name is None:
            self.name = 'star{}'.format(Star.index)
            Star.index += 1
        else:
            self.name = name
        self.c = color
        self.F = Vector(0, 0)
        self.a = Vector(0, 0)
        self.last_annotation = None

    def __eq__(self, other):
        return self.name == other.name

    def calculate_attractive_force(self, others):
        self.F = Vector(0, 0)
        for other in others:
            r2 = (self.pos - other.pos).get_l2norm()
            F = Star.G * self.m * other.m / r2
            self.F += F * (other.pos - self.pos).get_unit_vector()

    def calculate_acceleration(self):
        self.a = self.F / self.m

    def calculate_next_velocity(self):
        self.v = self.v + self.dt * self.a

    def calculate_displacement(self):
        self.delta = self.dt * self.v

    def infer(self, others, dt=0.1):
        self.dt = dt
        self.calculate_attractive_force(others)
        self.calculate_acceleration()
        self.calculate_next_velocity()
        self.calculate_displacement()

    def update_pos(self, canvas):
        self.pos += self.delta
        return self.show(canvas)

    def show(self, canvas):
        if self.last_annotation is not None:
            self.last_annotation.set_visible(False)
        self.last_annotation = canvas.annotate(self.name, (self.pos.x, self.pos.y))
        trace = canvas.scatter(self.pos.x, self.pos.y, color=self.c, marker='.')
        return trace

class StarSystem(object):
    def __init__(self, star_list, name='', dt=0.1, trace_len=None):
        self.star_list = star_list
        self.queue = []
        self.pos = []
        self.name = name
        self.dt = dt
        self.initial_state = ['{}: v0{}, x0{}'.format(s.name, s.v, s.pos) for s in star_list]
        if trace_len is None:
            self.trace_len = [100] * len(star_list)
        else:
            self.trace_len = trace_len
        for t_len in self.trace_len:
            self.queue.append(Queue(maxsize=t_len))

        self.fig = plt.figure(self.name)
        self.ax = self.fig.add_subplot(1, 1, 1)
        # self.ax.set_animated(True)
        self.ax.set_title('\n'.join(self.initial_state))
        self.ax.axis("equal")
        plt.grid(True)
        plt.ion()

    def infer(self):
        for s in self.star_list:
            s.infer([other for other in self.star_list if other != s], self.dt)

    def update_pos(self, canvas):
        for i, s in enumerate(self.star_list):
            q = self.queue[i]
            if q.full():
                p = q.get()
                # p.remove()
                p.set_visible(False)
            pos = s.update_pos(canvas)
            q.put(pos)
            valid_pos = [p for p in self.ax.get_children()
                         if (isinstance(p, matplotlib.collections.PathCollection) or isinstance(p, matplotlib.text.Text)) and
                             p.get_visible()]

            self.pos.append(valid_pos)
        plt.pause(0.001)

    def run(self, iterations=2000, saving=True):
        for i in range(iterations):
            self.infer()
            self.ax.set_xlabel('step{}'.format(i))
            self.update_pos(self.ax)

        if saving:
            ani = animation.ArtistAnimation(self.fig, self.pos, interval=20, repeat_delay=1000, blit=True)
            print('saving...')
            ani.save("{}.gif".format(self.name), writer='imagemagick')
            plt.show()

def run_solar_system():
    sun = Star(mass=40000, position=Vector(0, 0), velocity=Vector(0, 0), name='sun', color='r')
    earth = Star(mass=40, position=Vector(0, 10), velocity=Vector(2, 0), name='earth', color='g')
    s = StarSystem([sun, earth], name='s', trace_len=[100, 300])
    s.run()

def run_m_system():
    sun = Star(mass=40000, position=Vector(0, 0), velocity=Vector(0, 0), color='r')
    earth = Star(mass=4000, position=Vector(0, 10), velocity=Vector(1, 0), color='g')
    m = StarSystem([sun, earth], name='m', trace_len=[300, 100])
    m.run()

def run_twins_system():
    sun = Star(mass=40000, position=Vector(0, 0), velocity=Vector(2, 0), color='r')
    earth = Star(mass=40000, position=Vector(0, 5), velocity=Vector(-2, 0), color='g')
    o = StarSystem([sun, earth], name='o', dt=0.01, trace_len=[1000, 1000])
    o.run()

def run_balanced_three_body():
    # make sure GM = sqrt(3) * r * v**2
    sqrt3 = np.sqrt(3)
    m = 10000 * sqrt3
    r = 2.5
    v = 2
    star0 = Star(mass=m, position=Vector(0, r), velocity=Vector(v, 0), color='r')
    star1 = Star(mass=m, position=Vector(r * sqrt3/2, -r/2), velocity=Vector(-v/2, -v * sqrt3/2), color='g')
    star2 = Star(mass=m, position=Vector(-r * sqrt3/2, -r/2), velocity=Vector(-v/2, v * sqrt3/2), color='b')
    t = StarSystem([star0, star1, star2], name='balanced_three_body', dt=0.01, trace_len=[50, 50, 50])
    t.run(iterations=200, saving=True)

def run_three_body():
    # make some disturbance on the equation GM = sqrt(3) * r * v**2
    sqrt3 = np.sqrt(3)
    eps = 1e-1
    m = 10000 * sqrt3
    r = 2.5
    v = 2
    star0 = Star(mass=m, position=Vector(0, r), velocity=Vector(v + eps, 0), color='r')
    star1 = Star(mass=m, position=Vector(r * sqrt3 / 2, -r / 2), velocity=Vector(-v / 2, -v * sqrt3 / 2), color='g')
    star2 = Star(mass=m, position=Vector(-r * sqrt3 / 2, -r / 2), velocity=Vector(-v / 2, v * sqrt3 / 2), color='b')
    t = StarSystem([star0, star1, star2], name='three_body', dt=0.1, trace_len=[50, 50, 50])
    t.run(iterations=2000, saving=False)

if __name__ == '__main__':
    # run_solar_system()
    # run_m_system()
    # run_twins_system()
    # run_balanced_three_body()
    run_three_body()

