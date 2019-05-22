#coding:UTF-8
import numpy as np
import cv2
import math

class quadObject():
    def __init__(self, quadrilateral):
        '''

        :param rectangle:
        :param horizontal_style:
        '''
        self.quad = quadrilateral
        self.is_clockwise = False # False: uncertain; True: yes
        self.is_long_edge_first = False # False: uncertain; True: yes
        self.is_order_adjusted = False

        self.short_center_line = None
        self.long_center_line = None
        self.circumscribed_circle_diameter = None #近似外接圆直径
        self.angle_of_center_line = None #360: 不确定方向; -90< <=90: 正常方向范围
        self.center_point = None

        self.canvas_size = None
        self.color = None
        self.mask = None

        self._clockwise() # 转顺时针
        self._longEdgeFirst() # 第一条边变成长边
        self._orderAdjust()
        self._genLongShortCenterLine(ignore_ratio=1.5) # 长短边比大于ignore_ratio的生成长中心线，否认长中心线由两个中心点组成
        self._calAngleOfCenterLine() #　长中心线的与水平方向夹角
        self._center() #　长中心线的中心点

        # print('long_center_line', self.long_center_line)
        # print('angle', self.angle_of_center_line)
        # print('center', self.center_point)

    def _clockwise(self):
        '''
        第一个点保持不变，将self.rect变为顺时针
        :param rect:
        :return:
        '''
        if self.is_clockwise:
            return

        v1 = self.quad[1]-self.quad[0]
        v2 = self.quad[3]-self.quad[0]

        if np.cross(v1, v2) < 0:
            rect = self.quad[1].copy()
            self.quad[1] = self.quad[3]
            self.quad[3] = rect

        self.is_clockwise = True


    def _longEdgeFirst(self):
        '''
        顺时针下第一条边变为长边
        :return:
        '''
        assert self.is_clockwise

        if self.is_long_edge_first:
            return

        x = [0,0,0,0]
        y = [0,0,0,0]
        for i in range(4):
            x[i], y[i] = self.quad[i]

        square_edge_len = [0,0,0,0]

        for i in range(4):
            j = i%4
            k = (i+1)%4
            square_edge_len[i] = math.sqrt((x[j]-x[k])**2 + (y[j]-y[k])**2)

        avg_len1 = (square_edge_len[0]+square_edge_len[2])/2
        avg_len2 = (square_edge_len[1]+square_edge_len[3])/2

        # 第一条边变为长边
        if avg_len1 < avg_len2:
            p = self.quad[0].copy()
            self.quad[0] = self.quad[1]
            self.quad[1] = self.quad[2]
            self.quad[2] = self.quad[3]
            self.quad[3] = p

        self.is_long_edge_first = True

    def _orderAdjust(self):
        '''
        第一条有向边的角度在(-90, 90]内，
        :return:
        '''
        assert self.is_clockwise and self.is_long_edge_first
        A = self.quad[0]
        B = self.quad[1]
        AB = B-A
        if AB[0] < 0:
            p = self.quad[0].copy()
            q = self.quad[1].copy()
            self.quad[0] = self.quad[2]
            self.quad[1] = self.quad[3]
            self.quad[2] = p
            self.quad[3] = q
        elif AB[0] == 0:
            if AB[1] < 0:
                p = self.quad[0].copy()
                q = self.quad[1].copy()
                self.quad[0] = self.quad[2]
                self.quad[1] = self.quad[3]
                self.quad[2] = p
                self.quad[3] = q

        self.is_order_adjusted = True

    def _genLongShortCenterLine(self, ignore_ratio=1.5):
        '''
        沿长边的中心线
        :param ignore_ratio:
        :return:
        '''
        assert self.is_clockwise and self.is_long_edge_first and self.is_order_adjusted

        c1 = ((self.quad[0] + self.quad[1]) / 2).astype(np.int32)
        c2 = ((self.quad[1] + self.quad[2]) / 2).astype(np.int32)
        c3 = ((self.quad[2] + self.quad[3]) / 2).astype(np.int32)
        c4 = ((self.quad[3] + self.quad[0]) / 2).astype(np.int32)

        l1 = math.sqrt((c1[0]-c3[0])**2 + (c1[1]-c3[1])**2)
        l2 = math.sqrt((c2[0]-c4[0])**2 + (c2[1]-c4[1])**2)

        ratio = l2/l1

        if ratio <= ignore_ratio:
            center_point = ((c2 + c4) / 2).astype(np.int32)
            self.long_center_line = np.array([center_point, center_point]).astype(np.int32)
            self.short_center_line = np.array([center_point, center_point]).astype(np.int32)
        else:
            self.long_center_line = np.array([c4, c2]).astype(np.int32)
            self.short_center_line = np.array([c3, c1]).astype(np.int32)

        self.circumscribed_circle_diameter = math.sqrt(l1 ** 2 + l2 ** 2)
        return self.long_center_line, self.short_center_line, self.circumscribed_circle_diameter

    def _calAngleOfCenterLine(self):
        '''
        沿长边中心线的斜率
        :return:
        '''
        assert type(self.long_center_line) != type(None)

        x0, y0 = self.long_center_line[0]
        x1, y1 = self.long_center_line[1]

        if (self.long_center_line[0] == self.long_center_line[1]).all():
            self.angle_of_center_line = 360 #表示无方向
            # 考虑单个字可能是斜的，假设都应该接近水平
            # h = y1 - y0
            # w = x1 - x0
            #
            # if w != 0:
            #     angle = math.atan(h / w)
            #     self.angle_of_center_line = angle * 180 / np.pi
            #     self.angle_of_center_line = -self.angle_of_center_line

        else:
            h = y1-y0
            w = x1-x0

            if w != 0:
                angle = math.atan(h/w)
                self.angle_of_center_line = angle*180/np.pi
                self.angle_of_center_line = -self.angle_of_center_line

                # if self.angle_of_center_line <= 0:
                #     self.angle_of_center_line = -self.angle_of_center_line #+= 180
                # else:
                #     self.angle_of_center_line = 180-self.angle_of_center_line
            else:
                self.angle_of_center_line = 90

        return self.angle_of_center_line

    def _center(self):
        '''
        沿长边中心线的中心
        :return:
        '''
        assert type(self.long_center_line) != type(None)
        self.center_point = np.average(self.long_center_line, axis=0).astype(np.int32)
        return self.center_point

    def __distanceOfPointToStraightLine(self, point, line_start_point, line_end_point):
        P = np.array(list(point))
        A = np.array(list(line_start_point))
        B = np.array(list(line_end_point))

        AP = P - A
        AB = B - A

        if AB[0]**2+AB[1]**2 == 0:
            len_AP = math.sqrt(AP[0]**2+AP[1]**2)
            return len_AP

        r = (AP[0] * AB[0] + AP[1] * AB[1]) / (AB[0] ** 2 + AB[1] ** 2)
        AC = r * AB
        CP = AP - AC
        len_CP = math.sqrt(CP[0] ** 2 + CP[1] ** 2)
        return len_CP

    def __distanceOfPointToLineSegment(self, point, line_start_point, line_end_point):
        P = np.array(list(point))
        A = np.array(list(line_start_point))
        B = np.array(list(line_end_point))

        AP = P-A
        AB = B-A

        if AB[0]**2+AB[1]**2 == 0:
            len_AP = math.sqrt(AP[0]**2+AP[1]**2)
            return len_AP

        r = (AP[0]*AB[0]+AP[1]*AB[1])/(AB[0]**2+AB[1]**2)
        if r <= 0:
            len_AP = math.sqrt(AP[0]**2+AP[1]**2)
            return len_AP
        elif r >= 1:
            BP = P-B
            len_BP = math.sqrt(BP[0]**2+BP[1]**2)
            return len_BP
        else:
            AC = r*AB
            CP = AP-AC
            len_CP = math.sqrt(CP[0]**2+CP[1]**2)
            return len_CP

    def __signedRatioOfProjectLine(self, point, line_start_point, line_end_point):
        '''
        点P到有向线段AB投影，其投影线段与AB的比例(带方向)
        :param point:
        :param line_start_point:
        :param line_end_point:
        :return:
        '''
        P = np.array(list(point))
        A = np.array(list(line_start_point))
        B = np.array(list(line_end_point))

        AP = P - A
        AB = B - A

        if AB[0] ** 2 + AB[1] ** 2 == 0:
            # len_AP = math.sqrt(AP[0] ** 2 + AP[1] ** 2)
            # return len_AP
            return -1

        r = (AP[0] * AB[0] + AP[1] * AB[1]) / (AB[0] ** 2 + AB[1] ** 2)
        return r

    def genShiftMatrix(self, src_point, dst_point):
        shift_x, shift_y = src_point-dst_point
        M = np.array([[1, 0, shift_x],
                      [0, 1, shift_y],
                      [0, 0,       1]], dtype=np.float32)
        return M

    def genRotateMatrix(self, angle):
        '''
        angle是角度
        :param angle:
        :return:
        '''
        angle1 = angle*np.pi/180
        a = math.cos(angle1)
        b = math.sin(angle1)

        M = np.array([[ a,-b, 0],
                      [ b, a, 0],
                      [ 0, 0, 1]], dtype=np.float32)
        return M

    def genRotateAroundMatrix(self, center, angle):
        origin = np.array([0, 0], dtype=np.float32)
        M1 = self.genShiftMatrix(origin, center)
        M2 = self.genRotateMatrix(angle)
        M3 = self.genShiftMatrix(center, origin)
        M = M3.dot(M2.dot(M1))
        return M

    def rotateAroundCenter(self, center, angle):
        origin = np.array([0, 0], dtype=np.float32)
        M1 = self.genShiftMatrix(origin, center)
        M2 = self.genRotateMatrix(angle)
        M3 = self.genShiftMatrix(center, origin)
        M = M3.dot(M2.dot(M1))

        quad_proj = np.concatenate([self.quad.astype(np.float32), np.array([[1], [1], [1], [1]], dtype=np.float32)], axis=1)
        quad_proj = quad_proj.T
        rotated_quad_proj = M.dot(quad_proj)
        rotated_quad = rotated_quad_proj[:2, :].T.copy().astype(np.int32)
        new_quad_obj = quadObject(rotated_quad)
        return new_quad_obj, M

    def rotateByMatrix(self, M):
        quad_proj = np.concatenate([self.quad.astype(np.float32), np.array([[1], [1], [1], [1]], dtype=np.float32)], axis=1)
        quad_proj = quad_proj.T
        rotated_quad_proj = M.dot(quad_proj)
        rotated_quad = rotated_quad_proj[:2, :].T.copy().astype(np.int32)
        new_quad_obj = quadObject(rotated_quad)
        return new_quad_obj

    def shiftByOriginChange(self, dst_origin):
        origin = np.array([0, 0], dtype=np.float32)
        M = self.genShiftMatrix(origin, dst_origin)
        quad_proj = np.concatenate([self.quad.astype(np.float32), np.array([[1], [1], [1], [1]], dtype=np.float32)], axis=1)
        quad_proj = quad_proj.T
        shifted_quad = M.dot(quad_proj)
        shifted_quad = shifted_quad[:2, :].T.copy().astype(np.int32)
        new_quad_obj = quadObject(shifted_quad)
        return new_quad_obj, M

    def draw(self, canvas_size, color=255):
        '''
        给一多边形，按给定的颜色在给定大小的mask上填充多边形
        :param polygon: eg. [np.array([[100, 50], [500, 20], [600, 300], [130, 400]]).astype(np.int32)]
        :param canvas_size: eg. (800, 600)
        :param color: eg. 255 or (255, 255, 255)
        :return: mask of polygon with color
        '''

        polygon = self.quad
        self.canvas_size = canvas_size
        self.color = color

        assert (type(color) == type(int()) or (type(color) == type(tuple()) and len(color) == 3))
        assert (type(canvas_size) == type(tuple()))

        w, h = canvas_size

        if type(color) == type(int()):
            self.mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self.mask = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.fillPoly(self.mask, [polygon], color)

        return self.mask

    def drawLongCenterLine(self, canvas_size, color=255):
        '''

        :return:
        '''
        assert type(self.long_center_line) != type(None)
        assert (type(canvas_size) == type(tuple()))

        w, h = canvas_size

        if type(color) == type(int()):
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.line(mask, tuple(self.long_center_line[0]), tuple(self.long_center_line[1]), color, 2)
        return mask

    def erase(self):
        self.canvas_size = None
        self.color = None
        self.mask = None

    def intersect(self, quad_objB):
        '''
        计算mask_list中所有mask的相交面积，以及相交的图形mask. 注意只有当两者的color bitwise_and不为０时才会相交
        :param quadObjectB:
        :return:
        '''
        maskA = self.mask
        maskB = quad_objB.mask
        assert (type(maskA) != type(None) and type(maskB) != type(None))
        assert self.canvas_size == quad_objB.canvas_size
        intersect_mask = np.bitwise_and(maskA, maskB)
        nonzeros = np.nonzero(intersect_mask)
        intersect_area = len(nonzeros[0])
        return intersect_area, intersect_mask

    def union(self, quad_objB):
        '''
        计算mask_list中所有mask的相并面积，以及相并的图形mask.
        :param quad_objB:
        :return:
        '''
        maskA = self.mask
        maskB = quad_objB.mask
        assert (type(maskA) != type(None) and type(maskB) != type(None))
        assert self.canvas_size == quad_objB.canvas_size
        union_mask = np.bitwise_or(maskA, maskB)
        nonzeros = np.nonzero(union_mask)
        union_area = len(nonzeros[0])
        return union_area, union_mask

    # 半平面交法，待实现...

    def minDistanceTo(self, quad_objB):
        '''
        对称的
        两个框最近点的距离，作用：判断key和value之间的匹配
        0表示相交
        :param quad_objB:
        :return:
        '''
        intersect_area, intersect_mask = self.intersect(quad_objB)
        if intersect_area != 0:
            distance = 0
        else:
            d_list = []

            quadA = self.quad
            quadB = quad_objB.quad

            for i in range(len(quadA)):
                for j in range(len(quadB)):
                    P = quadA[i]
                    A = quadB[j%4]
                    B = quadB[(j+1)%4]
                    d = self.__distanceOfPointToLineSegment(P, A, B)
                    d_list.append(d)

            for i in range(len(quadB)):
                for j in range(len(quadA)):
                    P = quadB[i]
                    A = quadA[j%4]
                    B = quadA[(j+1)%4]
                    d = self.__distanceOfPointToLineSegment(P, A, B)
                    d_list.append(d)

            d_list = np.array(d_list)
            distance = np.min(d_list)
        return distance
    
    def probOfSameRowWith(self, quad_objB):
        '''
        非对称的
        与quad_objB是不是同一行，以quad_objB为参照
        当quad_objB的短边是一个点时：返回-1
        否则，当quad_objA的短边是一个点，quad_objB的短边不是一个点时:以quad_objA的中心为圆心，近似外接圆直径为直径，投影到quad_objB的短中心线上，计算公共部分的长度与quad_objB短中心线的长度的比值
        否则，当quad_objA和quad_objB的短边都不是一个点时：将quad_objA的短中心线投影到quad_objB的短中心线上，计算公共部分的长度与quad_objB短中心线的长度的比值
        :param quad_objB:
        :return:
        '''
        '''
        注意一个横框和一个竖框按此标准基本不是同行
        该函数只适用角度差在一定范围内的框
        '''
        short_center_lineA = self.short_center_line
        short_center_lineB = quad_objB.short_center_line

        flagA = np.sum(np.square(short_center_lineA[0] - short_center_lineA[1]))
        flagB = np.sum(np.square(short_center_lineB[0] - short_center_lineB[1]))

        if flagB == 0:
            return -1
        elif flagA == 0:
            A = self.center_point
            B = short_center_lineB[0]
            C = short_center_lineB[1]
            BC = C - B
            len_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
            r0 = self.__signedRatioOfProjectLine(A, B, C)
            r1 = r0 - self.circumscribed_circle_diameter / (2 * len_BC)
            r2 = r0 + self.circumscribed_circle_diameter / (2 * len_BC)
        else:
            r1 = self.__signedRatioOfProjectLine(short_center_lineA[0], short_center_lineB[0], short_center_lineB[1])
            r2 = self.__signedRatioOfProjectLine(short_center_lineA[1], short_center_lineB[0], short_center_lineB[1])

        r1 = np.clip(np.array(r1).astype(np.float32), 0, 1)
        r2 = np.clip(np.array(r2).astype(np.float32), 0, 1)

        # if r1 != 0 and r2 != 1: # 如果在内部，则概率为１
        #     prob = 1
        # else:
        prob = abs(r2 - r1)
        return prob

    def probOfSameColWith(self, quad_objB):
        '''
        非对称的
        与quad_objB是不是同一列，以quad_objB为参照
        当quad_objB的长边是一个点时：返回-1
        否则，当quad_objA的长边是一个点，quad_objB的长边不是一个点时:以quad_objA的中心为圆心，近似外接圆直径为直径，投影到quad_objB的长中心线上，计算公共部分的长度与quad_objB长中心线的长度的比值
        否则，当quad_objA和quad_objB的长边都不是一个点时：将quad_objA的长中心线投影到quad_objB的长中心线上，计算公共部分的的长度与quad_objB长中心线的长度的比值
        :param quad_objB:
        :return:
        '''
        '''
        注意一个横框和一个竖框按此标准基本不是同列
        该函数只适用角度差在一定范围内的框
        '''
        long_center_lineA = self.long_center_line
        long_center_lineB = quad_objB.long_center_line

        flagA = np.sum(np.square(long_center_lineA[0] - long_center_lineA[1]))
        flagB = np.sum(np.square(long_center_lineB[0] - long_center_lineB[1]))

        if flagB == 0:
            return -1
        elif flagA == 0:
            A = self.center_point
            B = long_center_lineB[0]
            C = long_center_lineB[1]
            BC = C - B
            len_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
            r0 = self.__signedRatioOfProjectLine(A, B, C)
            r1 = r0 - self.circumscribed_circle_diameter / (2 * len_BC)
            r2 = r0 + self.circumscribed_circle_diameter / (2 * len_BC)
        else:
            r1 = self.__signedRatioOfProjectLine(long_center_lineA[0], long_center_lineB[0], long_center_lineB[1])
            r2 = self.__signedRatioOfProjectLine(long_center_lineA[1], long_center_lineB[0], long_center_lineB[1])

        r1 = np.clip(np.array(r1).astype(np.float32), 0, 1)
        r2 = np.clip(np.array(r2).astype(np.float32), 0, 1)

        # if r1 != 0 and r2 != 1: # 如果在内部，则概率为１
        #     prob = 1
        # else:
        prob = abs(r2 - r1)
        return prob

    def probsOfAignedWith(self, quad_objB):
        '''
        非对称的
        与quad_objB是不是对齐，以quad_objB为参照
        当quad_objB的长边是一个点时：返回-1
        否则，当quad_objA的长边是一个点，quad_objB的长边不是一个点时:以quad_objA的中心为圆心，近似外接圆直径为直径，投影到quad_objB的长中心线上，然后见步骤B
        否则，当quad_objA和quad_objB的长边都不是一个点时：将quad_objA的长中心线投影到quad_objB的长中心线上，然后见步骤B
        步骤B: 假设quad_objB的长中心线为CD，quad_objA长中心线投影到CD上的点为E,F,计算E,F和C,D的距离，计算EF中点与CD中点的距离，如果距离大于quad_objB的短边，则不对齐，否则，对齐的概率为1-h1/h，
        其中，h1是前面计算出的距离，h是quad_objB的短边长
        :param quad_objB:
        :return:　左边对齐概率，　右边对齐概率，　中心对齐概率
        '''
        '''
        注意我们无法判断绝对的左边和右边，这里的左右是相对于quad_objB的第一条（长）边的左右顺序．
        因此，判断一组框是否为同边对齐，只需要判断它们是否同为左对齐或右对齐
        该函数只适用角度差在一定范围内的框
        '''
        short_center_lineB = self.short_center_line

        long_center_lineA = self.long_center_line
        long_center_lineB = quad_objB.long_center_line

        flagA = np.sum(np.square(long_center_lineA[0] - long_center_lineA[1]))
        flagB = np.sum(np.square(long_center_lineB[0] - long_center_lineB[1]))

        A = self.center_point
        B = long_center_lineB[0]
        C = long_center_lineB[1]
        BC = C - B
        len_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

        if flagB == 0:
            return -1
        elif flagA == 0:
            r0 = self.__signedRatioOfProjectLine(A, B, C)
            r1 = r0 - self.circumscribed_circle_diameter / (2 * len_BC)
            r2 = r0 + self.circumscribed_circle_diameter / (2 * len_BC)
            rc = (r1+r2)/2
        else:
            r1 = self.__signedRatioOfProjectLine(long_center_lineA[0], B, C)
            r2 = self.__signedRatioOfProjectLine(long_center_lineA[1], B, C)
            rc = (r1+r2)/2

        prob_left_align = 0
        prob_center_align = 0
        prob_right_align = 0

        short_center_line_len = np.sqrt(np.sum(np.square(short_center_lineB[0]-short_center_lineB[1])))
        standard_len = short_center_line_len / len_BC

        if min(abs(r1), abs(r2)) < standard_len:
            if abs(r1) <= abs(r2) and r2 >= r1:
                prob_left_align = 1 - min(abs(r1), abs(r2))/standard_len
            elif abs(r1) >= abs(r2) and r1 >= r2:
                prob_left_align = 1 - min(abs(r1), abs(r2)) / standard_len

        if abs(0.5-rc) < standard_len:
            prob_center_align = 1 - abs(0.5-rc)/standard_len

        if min(abs(1-r1), abs(1-r2)) < standard_len:
            if abs(1-r1) <= abs(1-r2) and r2 <= r1:
                prob_right_align = 1 - min(abs(1-r1), abs(1-r2))/standard_len
            elif abs(1-r2) <= abs(1-r1) and r1 <= r2:
                prob_right_align = 1 - min(abs(1 - r1), abs(1 - r2)) / standard_len

        return prob_left_align, prob_center_align, prob_right_align

        # if min(abs(r1), abs(r2), abs(1-r1), abs(1-r1)) <= standard_len:
        #     prob_left_align = 1-min(abs(r1), abs(r2), abs(1-r1), abs(1-r2))
        #
        # if abs(r1) <= standard_len:
        #     prob_left_align = 1-abs(r1)
        # elif abs(len_BC-r1) <= standard_len:
        #     prob_right_align = 1-abs(len_BC-r1)

    # 待验证
    def probRelativePositionTo(self, quad_objB, is_standard_coordinate_system=True):
        '''
        quad_objA相对于quad_objB的位置关系，有上中下，左中右．is_standard_coordinate_system为True时，参考坐标系为水平垂直坐标系，为False时，未定义
        :param quad_objB:
        :param is_standard_coordinate_system:
        :return:
        '''

        # 左中右关系
        A = quad_objB.center_point + np.array([1, 0], dtype=np.int32)
        B = quad_objB.center_point + np.array([-1, 0], dtype=np.int32)
        # A,B的四个顶点都投影到水平线上，水平线的中心为B的中心
        horizontal_project_ratiosB = [self.__signedRatioOfProjectLine(quad, A, B) for quad in quad_objB.quad]
        horizontal_project_ratiosA = [self.__signedRatioOfProjectLine(quad, A, B) for quad in self.quad]

        left_B = np.min(horizontal_project_ratiosB)
        right_B = np.max(horizontal_project_ratiosB)

        left_A = np.min(horizontal_project_ratiosA)
        right_A = np.max(horizontal_project_ratiosA)

        len_A_left_to_B = max(0, left_B - left_A)
        len_A_right_to_B = max(0, right_A-right_B)
        # len_A_in_B = (right_A-left_A)-len_A_left_to_B-len_A_right_to_B

        prob_A_left_to_B = len_A_left_to_B / (right_A-left_A)
        prob_A_right_to_B = len_A_right_to_B / (right_A-left_A)
        prob_A_in_B_horizontal = 1-prob_A_left_to_B-prob_A_right_to_B

        # 上中下关系
        A = quad_objB.center_point + np.array([0, -1], dtype=np.int32)
        B = quad_objB.center_point + np.array([0, 1], dtype=np.int32)
        # A,B的四个顶点都投影到垂直上，垂直线的中心为B的中心
        vertical_project_ratiosB = [self.__signedRatioOfProjectLine(quad, A, B) for quad in quad_objB.quad]
        vertical_project_ratiosA = [self.__signedRatioOfProjectLine(quad, A, B) for quad in self.quad]

        lower_B = np.min(vertical_project_ratiosB)
        upper_B = np.max(vertical_project_ratiosB)

        lower_A = np.min(vertical_project_ratiosA)
        upper_A = np.max(vertical_project_ratiosA)

        len_A_lower_than_B = max(0, lower_B-lower_A)
        len_A_upper_than_B = max(0, upper_A-upper_B)

        prob_A_lower_than_B = len_A_lower_than_B / (upper_A-lower_A)
        prob_A_upper_than_B = len_A_upper_than_B / (upper_A-lower_A)
        prob_A_in_B_vertical = 1-prob_A_lower_than_B-prob_A_upper_than_B

        probs = np.array([[prob_A_left_to_B, prob_A_in_B_horizontal, prob_A_right_to_B],
                        [prob_A_lower_than_B, prob_A_in_B_vertical, prob_A_upper_than_B]], dtype=np.float32)

        return probs


if __name__ == '__main__':

    polygon1 = np.array([[100, 50], [500, 20], [600, 300], [130, 400]]).astype(np.int32)
    polygon2 = np.array([[50, 60], [480, 28], [700, 270], [100, 420]]).astype(np.int32)
    polygon2 = np.array([[100, 420], [700, 270], [480, 28], [50, 60]]).astype(np.int32)
    ele1 = quadObject(polygon1)
    ele2 = quadObject(polygon2)
    ele1.draw((800,600), (255, 0, 255)) #(255,0,0))
    ele2.draw((800,600), (255, 0, 255)) #(255,0,0))

    M = ele1.genRotateAroundMatrix(ele1.center_point, -ele1.angle_of_center_line)
    ele3 = ele1.rotateByMatrix(M)
    ele3.draw((800, 600), (255, 0, 255))

    from tools import rotateImageByMatrix
    mask3_1 = rotateImageByMatrix(ele1.mask, M)

    cv2.imshow('mask3_1', mask3_1)
    cv2.imshow('mask3', ele3.mask)
    cv2.waitKey(0)


#     # area, mask = ele1.intersect(ele2)
#     # cv2.imshow('intersect', mask)
#     # print(area)
#     # area, mask = ele1.union(ele2)
#     # print(area)
#     # cv2.imshow('union', mask)
#     # cv2.waitKey(0)
#     # quad = ele1.rotateAroundCenter((0,0), 10)
#     ele3, M = ele1.rotateAroundCenter(ele1.center_point, 10)
#     #ele3 = quadObject(quad)
#     ele3.draw((800, 600), (255, 0, 255))
#
#     cv2.imshow('mask1', ele1.mask)
#     cv2.imshow('mask3', ele3.mask)
#     cv2.waitKey(0)
#
#     ele4, M = ele1.shiftByOriginChange((100, 200))
#     #ele4 = quadObject(quad)
#     ele4.draw((800, 600), (255, 0, 255))
#     cv2.imshow('mask4', ele4.mask)
#     cv2.waitKey(0)
#
#     # quad = ele1.rotate(-30)
#     #
#     # ele3 = quadObject(quad)
#     # ele3.draw((800, 600), (255, 0, 255))
#     # cv2.imshow('mask3', ele3.mask)
#     # cv2.waitKey(0)
#
#     # cv2.imshow('ele', ele2.mask)
#     # cv2.waitKey(0)
#
#     # mask = ele.genPolyMask((800,600), 255)
#     # cv2.imshow('mask', mask)
#     # cv2.waitKey(0)
# # mask = np.zeros((800, 600), dtype=np.uint8)
# # cv2.fillPoly(mask, [polygon1], 255)
# #
# #
# # cv2.imshow('mask', mask)
# # cv2.waitKey(0)
