from numpy import array, dot, cross, linalg
from math import sqrt, sin, acos, atan2
from pykep import epoch, lambert_problem, fb_vel, DAY2SEC, RAD2DEG, DEG2RAD
from pykep.planet import keplerian


class flyby:

    # pykep.planet.keplerian(when,orbital_elements, mu_central_body, mu_self,radius, safe_radius [, name = ‘unknown’])
    # orbital_elements: a sequence of six containing a,e,i,W,w,M (SI units, i.e. meters and radiants)
    #    a is the semi-major axis, always a positive quantity.
    #    e is the eccentricity, non-negative
    #    i is the incliniation
    #    W is the longitude of the ascending node, undefined in an equatorial orbit
    #    w is the argument of perigee, undefined in a circular orbit
    AU = 9832684544
    MU_STAR = 1.17233279483249E+18
    G = 6.67408e-11

    ksp_planet = {
        'moho': keplerian(epoch(0, "mjd2000"), (5263138304, 0.200000002980232, 7 * DEG2RAD, 70 * DEG2RAD, 15 * DEG2RAD, 3.14000010490417), MU_STAR, 168609378654.509, 250000, 250000, 'Moho'),
        'eve': keplerian(epoch(0, "mjd2000"), (9832684544, 0.00999999977648258, 2.09999990463257 * DEG2RAD, 15 * DEG2RAD, 0 * DEG2RAD, 3.14000010490417), MU_STAR, 8171730229210.87, 700000, 790000, 'Eve'),
        'kerbin': keplerian(epoch(0, "mjd2000"), (13599840256, 0, 1e-99, 0 * DEG2RAD, 0 * DEG2RAD, 3.14000010490417), MU_STAR, 3531600000000, 600000, 670000, 'Kerbin'),
        'duna': keplerian(epoch(0, "mjd2000"), (20726155264, 0.0509999990463257, 0.0599999986588955 * DEG2RAD, 135.5 * DEG2RAD, 0 * DEG2RAD, 3.14000010490417), MU_STAR, 301363211975.098, 320000, 370000, 'Duna'),
        'dres': keplerian(epoch(0, "mjd2000"), (40839348203, 0.145, 5 * DEG2RAD, 280 * DEG2RAD, 90 * DEG2RAD, 3.14000010490417), MU_STAR, 21484488600, 138000, 138000, 'Dres'),
        'jool': keplerian(epoch(0, "mjd2000"), (68773560320, 0.0500000007450581, 1.30400002002716 * DEG2RAD, 52 * DEG2RAD, 0 * DEG2RAD, 0.100000001490116), MU_STAR, 282528004209995, 6000000, 6200000, 'Jool'),
        'eeloo': keplerian(epoch(0, "mjd2000"), (90118820000, 0.26, 6.15 * DEG2RAD, 50 * DEG2RAD, 260 * DEG2RAD, 3.14000010490417), MU_STAR, 74410814527.0496, 210000, 210000, 'Eeloo')
    }

    def __init__(self, flight_plan, travel_days, windows, ignore_last=False, orbit_alt=100000, days=1, multi_revs=5):

        print(flyby.ksp_planet['moho'].eph(epoch(1000,"mjd2000")))

        # print(flyby.ksp_planet)
        self.travel_days = travel_days.copy()
        self.windows = windows.copy()
        self.orbit_alt = orbit_alt
        self.ignore_last = ignore_last
        self.dim = len(flight_plan)
        self.flight_plan = flight_plan.copy()
        self.days = days
        self.multi_revs = multi_revs

        self.planets = []
        self.mu_sun = flyby.MU_STAR
        for i in flight_plan:
            self.planets.append(self.ksp_planet[i])

        self.x = [None] * self.dim
        self.t = [None] * self.dim
        self.r = [None] * self.dim
        self.v = [None] * self.dim
        self.vo = [None] * self.dim
        self.vi = [None] * self.dim
        self.f = [None] * self.dim
        self.f_all = 0
        self.li_sol = []
        self.l = []

    def get_bounds(self):
        return ([0.0]*self.dim, self.windows)

    def get_name(self):
        return "Flyby"

    def fitness(self, x):
        self.x = x
        # calculate the times
        self.t[0] = self.travel_days[0] + self.x[0]
        for i in range(1, self.dim):
            self.t[i] = self.days * self.t[i - 1] + self.travel_days[i] + self.x[i]

        # calculate the state vectors of planets
        for i in range(self.dim):
            self.r[i], self.v[i] = self.planets[i].eph(
                epoch(self.t[i], "mjd2000"))

        # calculate the solutions of the two Lambert transfers
        self.l = []
        n_sols = []
        for i in range(self.dim - 1):
            self.l.append(lambert_problem(self.r[i], self.r[i + 1], (self.t[i + 1] - self.t[i]) * DAY2SEC, self.mu_sun, False, self.multi_revs))
            n_sols.append(self.l[i].get_Nmax() * 2 + 1)

        # perform the dV calculations
        mu0 = self.planets[0].mu_self
        rad0 = self.planets[0].radius + self.orbit_alt
        mu1 = self.planets[-1].mu_self
        rad1 = self.planets[-1].radius + self.orbit_alt

        k = 1
        for i in range(self.dim - 1):
            k = k * n_sols[i]

        vot = [0] * self.dim
        vit = [0] * self.dim
        ft = [0] * self.dim
        self.f_all = 1.0e10

        for kk in range(k):
            d = kk
            li = []
            for j in range(self.dim - 1):
                d, b = divmod(d, n_sols[j])
                li.append(b)

            vot[0] = array(self.l[0].get_v1()[li[0]]) - self.v[0]
            ft[0] = sqrt(dot(vot[0], vot[0]) + 2 * mu0 / rad0) - sqrt(1 * mu0 / rad0)

            if ft[0] > self.f_all:
                continue

            for i in range(1, self.dim - 1):
                vit[i] = array(self.l[i - 1].get_v2()[li[i - 1]]) - self.v[i]
                vot[i] = array(self.l[i].get_v1()[li[i]]) - self.v[i]
                ft[i] = fb_vel(vit[i], vot[i], self.planets[i])

            vit[-1] = array(self.l[-1].get_v2()[li[-1]]) - self.v[-1]
            ft[-1] = sqrt(dot(vit[-1], vit[-1]) + 2 * mu1 / rad1) - sqrt(2 * mu1 / rad1)

            ft_all = sum(ft)
            if self.ignore_last:
                ft_all = ft_all - ft[-1]

            if ft_all < self.f_all:
                self.f_all = ft_all
                self.vi = vit.copy()
                self.vo = vot.copy()
                self.f = ft.copy()
                self.li_sol = li.copy()

        # check and set cost of negative altitude (using safe_radius)
        res = self.f_all
        for i in range(1, self.dim - 1):
            ta = acos(dot(self.vi[i], self.vo[i]) / sqrt(dot(self.vi[i], self.vi[i])) / sqrt(dot(self.vo[i], self.vo[i])))
            alt = (self.planets[i].mu_self / dot(self.vi[i], self.vi[i]) * (1 / sin(ta / 2) - 1)) - self.planets[i].safe_radius
            if alt < 0:
                res = res - alt

        # return the total fuel cost
        # print(x)
        # print(res)
        return [res]

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

    def plot_trajectory(self):
        from matplotlib.pyplot import show, axes
        from pykep.orbit_plots import plot_planet, plot_lambert

        ax = axes(projection='3d')
        ax.scatter([0], [0], [0], color='y')

        colors = {'moho': '#7B7869',
                  'eve': '#BB91A1',
                  'kerbin': '#0000FF',
                  'duna': '#E27B58',
                  'dres': '#A49B72',
                  'jool': '#C88B3A',
                  'eeloo': '#333333'}

        d_max = 0
        for i in range(self.dim):
            r, v = self.planets[i].eph(epoch(self.t[i], "mjd2000"))
            d = dot(r, r)
            if d > d_max:
                d_max = d

            p = keplerian(epoch(self.t[i], "mjd2000"),
                          self.planets[i].osculating_elements(epoch(self.t[i], "mjd2000")),
                          self.planets[i].mu_central_body,
                          self.planets[i].mu_self,
                          self.planets[i].radius,
                          self.planets[i].safe_radius,
                          self.flight_plan[i])
            # print(p)
            plot_planet(p, epoch(self.t[i], "mjd2000"), units=flyby.AU, color=colors[self.flight_plan[i]], axes=ax)
            if i != self.dim - 1:
                plot_lambert(self.l[i], sol=self.li_sol[i], units=flyby.AU, color='c', axes=ax)

        d_max = 1.2 * sqrt(d_max) / flyby.AU
        ax.set_xlim(-d_max, d_max)
        ax.set_ylim(-d_max, d_max)
        ax.set_zlim(-d_max, d_max)

        show()

    def print_transx(self):

        print("Date of %8s departue : " %
              self.flight_plan[0], epoch(self.t[0], "mjd2000").mjd2000)
        for i in range(1, self.dim - 1):
            print("Date of %8s encounter: " %
                  self.flight_plan[i], epoch(self.t[i], "mjd2000").mjd2000)
        print("Date of %8s arrival  : " %
              self.flight_plan[-1], epoch(self.t[-1], "mjd2000").mjd2000)
        print("")

        for i in range(self.dim - 1):
            print("Transfer time from %8s to %8s:" % (self.flight_plan[i], self.flight_plan[i + 1]),
                  self.t[i + 1] - self.t[i], " days")

        print("Total mission duration:                 ",
              self.t[-1] - self.t[0], " days")
        print("")
        print("")

        fward = [0] * self.dim
        plane = [0] * self.dim
        oward = [0] * self.dim
        for i in range(self.dim):
            fward[i] = self.v[i] / linalg.norm(self.v[i])
            plane[i] = cross(self.v[i], self.r[i])
            plane[i] = plane[i] / linalg.norm(plane[i])
            oward[i] = cross(plane[i], fward[i])

        print("TransX escape plan -  %8s escape" % self.flight_plan[0])
        print("=" * 70)
        print("MJD2000:                       %10.8f " % epoch(self.t[0], "mjd2000").mjd2000)
        print("Prograde:             %10.4f m/s" % round(dot(fward[0], self.vo[0]), 4))
        print("Outward:              %10.4f m/s" % round(dot(oward[0], self.vo[0]), 4))
        print("Plane:                %10.4f m/s" % round(dot(plane[0], self.vo[0]), 4))
        print("Hyp. excess velocity: %10.4f m/s" % round(linalg.norm(self.vo[0]), 4))
        print("%8s escape burn: %10.4f m/s" % (self.flight_plan[0], round(self.f[0], 4)))

        c3 = dot(self.vo[0], self.vo[0]) / 1000000
        dha = atan2(self.vo[0][2], sqrt(self.vo[0][0] * self.vo[0][0] + self.vo[0][1] * self.vo[0][1])) * RAD2DEG
        rha = atan2(self.vo[0][1], self.vo[0][0]) * RAD2DEG

        print("------------------GMAT------------------")
        print("GMAT MJD2000:         ", epoch(self.t[0], "mjd2000").mjd2000)
        print("GMAT OutgoingC3:      ", c3)
        print("GMAT OutgoingRHA:     ", rha)
        print("GMAT OutgoingDHA:     ", dha)
        print("")

        for i in range(1, self.dim - 1):
            vx = dot(fward[i], self.vo[i])
            vy = dot(oward[i], self.vo[i])
            vz = dot(plane[i], self.vo[i])
            mu = self.planets[i].mu_self
            rad = self.planets[i].radius
            print("%8s encounter" % self.flight_plan[i])
            print("=" * 70)
            print("MJD2000:             %10.4f " % round(epoch(self.t[i], "mjd2000").mjd2000, 4))
            print("Solution number:     %10d " % (1 + self.li_sol[i - 1]))
            print("Approach velocity:   %10.4f m/s" % round(linalg.norm(self.vi[i]), 4))
            print("Departure velocity:  %10.4f m/s" % round(linalg.norm(self.vo[i]), 4))
            print("Outward angle:       %10.4f deg" % round(atan2(vy, vx) * RAD2DEG, 4))
            print("Inclination:         %10.4f deg" % round(atan2(vz, sqrt(vx * vx + vy * vy)) * RAD2DEG, 4))

            a = - mu / dot(self.vi[i], self.vi[i])
            ta = acos(dot(self.vi[i], self.vo[i]) / linalg.norm(self.vi[i]) / linalg.norm(self.vo[i]))
            e = 1 / sin(ta / 2)
            rp = a * (1 - e)
            alt = (rp - rad) / 1000

            print("Turning angle:       %10.4f deg" % round(ta * RAD2DEG, 4))
            print("Periapsis altitude:  %10.4f km " % round(alt, 4))
            print("dV needed:           %10.4f m/s" % round(self.f[i], 4))
            print("------------------GMAT------------------")
            print("GMAT MJD2000:         ", epoch(self.t[i], "mjd2000").mjd2000)
            print("GMAT RadPer:          ", rp / 1000)

            c3 = dot(self.vi[i], self.vi[i]) / 1000000
            dha = atan2(-self.vi[i][2], sqrt(self.vi[i][0] * self.vi[i][0] + self.vi[i][1] * self.vi[i][1])) * RAD2DEG
            rha = atan2(-self.vi[i][1], -self.vi[i][0]) * RAD2DEG
            if rha < 0:
                rha = 360 + rha

            print("GMAT IncomingC3:      ", c3)
            print("GMAT IncomingRHA:     ", rha)
            print("GMAT IncomingDHA:     ", dha)

            e = cross([0, 0, 1], -self.vi[i])
            e = e / linalg.norm(e)
            n = cross(-self.vi[i], e)
            n = n / linalg.norm(n)
            h = cross(self.vi[i], self.vo[i])
            b = cross(h, -self.vi[i])
            b = b / linalg.norm(b)
            sinb = dot(b, e)
            cosb = dot(b, -n)
            bazi = atan2(sinb, cosb) * RAD2DEG
            if bazi < 0:
                bazi = bazi + 360
            print("GMAT IncomingBVAZI:   ", bazi)

            c3 = dot(self.vo[i], self.vo[i]) / 1000000
            dha = atan2(self.vo[i][2], sqrt(self.vo[i][0] * self.vo[i][0] + self.vo[i][1] * self.vo[i][1])) * RAD2DEG
            rha = atan2(self.vo[i][1], self.vo[i][0]) * RAD2DEG
            if rha < 0:
                rha = 360 + rha

            print("GMAT OutgoingC3:      ", c3)
            print("GMAT OutgoingRHA:     ", rha)
            print("GMAT OutgoingDHA:     ", dha)

            e = cross([0, 0, 1], self.vo[i])
            e = e / linalg.norm(e)
            n = cross(self.vo[i], e)
            n = n / linalg.norm(n)
            h = cross(self.vi[i], self.vo[i])
            b = cross(h, self.vo[i])
            b = b / linalg.norm(b)
            sinb = dot(b, e)
            cosb = dot(b, -n)
            bazi = atan2(sinb, cosb) * RAD2DEG
            if bazi < 0:
                bazi = bazi + 360
            print("GMAT OutgoingBVAZI:   ", bazi)

            print("")

        print("%8s arrival" % self.flight_plan[-1])
        print("=" * 70)
        print("MJD2000:              %10.4f    " % round(epoch(self.t[-1], "mjd2000").mjd2000, 4))
        print("Solution number:      %10d " % (1 + self.li_sol[-1]))
        print("Hyp. excess velocity: %10.4f m/s" % round(sqrt(dot(self.vi[-1], self.vi[-1])), 4))
        print("Orbit insertion burn  %10.4f m/s - C3 = 0" % round(self.f[-1], 4))

        c3 = dot(self.vi[-1], self.vi[-1]) / 1000000
        dha = atan2(self.vi[-1][2], sqrt(self.vi[-1][0] * self.vi[-1][0] + self.vi[-1][1] * self.vi[-1][1])) * RAD2DEG
        rha = atan2(self.vi[-1][1], self.vi[-1][0]) * RAD2DEG

        print("------------------GMAT------------------")
        print("GMAT MJD2000:         ", epoch(self.t[-1], "mjd2000").mjd2000)
        print("GMAT IncomingC3:      ", c3)
        print("GMAT IncomingRHA:     ", rha)
        print("GMAT IncomingDHA:     ", dha)
        print("")

        print("=" * 40)
        print("Total fuel cost:     %10.4f m/s" % round(self.f_all, 4))
