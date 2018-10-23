#!/usr/bin/env python
from sense_and_drive import *
import unittest
import numpy as np
import time
import math

class InformalTests:
    @staticmethod
    def test_intersect():
        for angle in range(360):
            for dist in range(80):
                #laser_reading = (dist/10.0, math.pi/2)#laser_angle(300))
                laser_reading = (dist/10.0, angle/(2*math.pi))#laser_angle(300))
                laser_xy = pol_to_cart(laser_reading)
                print(laser_xy)
                cells = []
                for r in range(GRID_Y):
                    for c in range(GRID_X):
                        i = r*GRID_X + c
                        cells.append((i,
                            GRID_CELLS[i].intersects_line(((0, 0), laser_xy)),
                            GRID_CELLS[i].intersects_point(*laser_xy)))
                draw_cells(cells)
                time.sleep(0.01)
        '''
        aabb = AABB(-1, 0, 0, 1)
        line = ((1, -1), (-1, 1))
        print(aabb.intersects_point(0.5, 0.5))
        print(aabb.intersects_point(-2, 0))
        print(aabb.intersects_line(line))
        '''

    @staticmethod
    def test_draw_cells():
        t, f = True, False
        draw_cells([(f,f), (f,f), (f,f), (f,f)], 2, 2)
        draw_cells([(t,t)], 1, 1)
#InformalTests.test_draw_cells()


class SenseAndDriveTests(unittest.TestCase):
    def test_smallest_angle_between(self):
        sab = lambda a,b,res: np.isclose(smallest_angle_between(a,b), res)
        pi = math.pi

        self.assertTrue(sab(tau-0.3, 0.4, 0.7))
        self.assertTrue(sab(0.4, tau-0.3, -0.7))
        self.assertTrue(sab(0.7, 0.9, 0.2))
        self.assertTrue(sab(0.9, 0.7, -0.2))
        self.assertTrue(sab(tau+0.9, 0.7, -0.2))
        self.assertTrue(sab(tau+0.9, -tau+0.7, -0.2))
        self.assertTrue(sab(0, 0.4, 0.4))
        self.assertTrue(sab(0, 4.5, 4.5-tau))
        self.assertTrue(sab(4.5, 0, tau-4.5))
        self.assertTrue(sab(0, 0, 0))
        # does not work with negative inputs

    def test_lines_intersect(self):
        def test_lines(a, b, c, d, should_intersect):
            ''' test permutations start/end points for each line '''
            self.assertEquals(lines_intersect((a, b), (c, d)), should_intersect)
            self.assertEquals(lines_intersect((b, a), (d, c)), should_intersect)
            self.assertEquals(lines_intersect((a, b), (d, c)), should_intersect)
            self.assertEquals(lines_intersect((b, a), (c, d)), should_intersect)

            self.assertEquals(lines_intersect((c, d), (a, b)), should_intersect)
            self.assertEquals(lines_intersect((d, c), (b, a)), should_intersect)
            self.assertEquals(lines_intersect((d, c), (a, b)), should_intersect)
            self.assertEquals(lines_intersect((c, d), (b, a)), should_intersect)

        test_lines((0, 0), (0, 1),  (0, 0), (0, 1), True)
        test_lines((0, 0), (1, 1),  (1, 1), (2, 2), True)
        test_lines((0, 0), (1, 1),  (0, 0), (1, 1), True)
        test_lines((0, 0), (1, 1),  (0, 0), (2, 2), True)
        test_lines((0, 0), (0, 1),  (0, 2), (0, 3), False)
        test_lines((0, 0), (1, 0),  (2, 0), (3, 0), False)
        test_lines((0, 0), (1, 1),  (2, 2), (3, 3), False)
        test_lines((0, 0), (-1, -1),  (-2, -2), (-3, -3), False)
        ##

        test_lines((0, 0), (0, 1),  (1, 0), (1, 1), False)
        test_lines((0, 0), (1, 0),  (0, 1), (1, 1), False)
        test_lines((0, 0), (0, 1),  (0, 0), (0, 2), True)
        test_lines((0, 0), (2, 0),  (0, 0), (2, 0), True)
        test_lines((0, 0), (2, 0),  (0, -1), (2, 1), True)
        test_lines((1, 1), (-1, -1), (0, 0), (3, 0), True)
        test_lines((0, 0), (1, -1), (0, 0), (-1, -1), True)
        test_lines((2, 2), (-1, -1), (3, 0), (5, 0), False)
        test_lines((-1, 1), (-2, -1), (1, 1), (2, -1), False)
        test_lines((-1, 0), (-1, 0), (1, -1), (1, -1), False)
        test_lines((0, 0), (0, 0), (0, 0), (0, 0), True)
        test_lines((0, 0), (0, 0), (-1, -1), (1, 1), True)
        test_lines((-1, 0), (-1, 0), (-1, -1), (1, 1), False)

    def test_intersect_point(self):
        def test_point(minx, miny, maxx, maxy, point_x, point_y, expected):
            ''' creates aabb box and tests if point is inside it '''
            test_box = AABB(minx, miny, maxx, maxy)
            self.assertEquals(test_box.intersects_point(point_x, point_y), expected)
        test_point(0, 0, 1, 1, 0.5, 0.5, True)
        test_point(0, 0, 1, 1, 1, 0.5, True)
        test_point(0, 0, 1, 1, 1, 1, True)
        test_point(0, 0, 1, 1, 5, 3, False)

    def test_Gauss(self):
        self.assertEquals(1, Gauss_num_sols(np.matrix([
            [0, -2,  1,  3],
            [2, 0,  -2,  7],
            [-5, 10, -1, -9]
        ], dtype=float)))
        self.assertEquals(1, Gauss_num_sols(np.matrix([
            [0, -2, 3, 3],
            [2, -1, -2, 6],
            [-5, -2, -1, -3]
        ], dtype=float)))
        self.assertEquals(0, Gauss_num_sols(np.matrix([
            [ 1, -2,  1,  3],
            [ 2, -4, -2,  7],
            [-5, 10, -1, -9]
        ], dtype=float)))
        self.assertEquals('inf', Gauss_num_sols(np.matrix([
            [-2, 2, -1, 4],
            [3, 2, 2, -1],
            [-1, -4, -1, -3]
        ], dtype=float)))


    def test_get_end_point(self):
        gep = lambda a,b,c,d,res_x,res_y: (
                np.isclose(get_end_point(a,b,c,d)[0],res_x) and
                np.isclose(get_end_point(a,b,c,d)[1],res_y))
        a = 0.6435011088 # radians between 0 and atan(4/3)
        self.assertTrue(gep((0, 0), (7, 0), 1.3, 1.3-math.pi/2.0, 7, 0))
        self.assertTrue(gep((0, 0), (-7, 0), 1.3, 1.3+math.pi/2.0, -7, 0))
        self.assertTrue(gep((0, 0), (0, 12), 0.45, 0.45, 0, 12))
        self.assertTrue(gep((0, 0), (3, 4), 0, 2*math.pi-a, 3, 4))
        self.assertTrue(gep((0, 0), (-3, 4), 0, a, -3, 4))
        self.assertTrue(gep((7, 4), (14, 4), math.pi, math.pi/2.0, 7, 0))
        self.assertTrue(gep((8, 4), (1, 4), math.pi/2.0, math.pi, -7, 0))
        self.assertTrue(gep((16.4, 7), (16.4, 9.5), 0.95, 0.95, 0, 2.5))
        self.assertTrue(gep((2.5, 1), (5.5, 5), 0, 2*math.pi-a, 3, 4))
        self.assertTrue(gep((2.5, 1), (-0.5, 5), 0, a, -3, 4))

    #def test_draw_cells(self):

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SenseAndDriveTests)
    unittest.TextTestRunner(verbosity=2).run(suite)

