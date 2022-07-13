import numpy as np
from k_means_constrained import KMeansConstrained
from group import Group
import sys
from copy import deepcopy, copy
from math import *


def two_way(A, B, groups_):

    indicator = 0

    # graph: if G > 0 redo the process
    while True:

        gain_buffer = []
        gain = 0
        x_star = []
        y_star = []

        #compute D values for A and B (block1)
        for p1 in A.outerPoints:
            internal_cost = sum(
                [1 if p2.group == A.name else 0 for p2 in p1.connected_points])
            external_cost = sum(
                [1 if p2.group == B.name else 0 for p2 in p1.connected_points])
            p1.Dvalue = external_cost - internal_cost

        for p1 in B.outerPoints:
            internal_cost = sum(
                [1 if p2.group == B.name else 0 for p2 in p1.connected_points])
            external_cost = sum(
                [1 if p2.group == A.name else 0 for p2 in p1.connected_points])
            p1.Dvalue = external_cost - internal_cost

        #create copy to avoid modifying the original sets
        A_copy = copy(A.outerPoints)        
        B_copy = copy(B.outerPoints)

        # flowchart: for p <-1 to n
        for k in range(min(len(A.outerPoints), len(B.outerPoints))):

            #select a_ and b_ such that gain = a_.Dvalue + b_.Dvalue - 2cost(a_,b_) is maximum
            all_dvalue_combinations = [(p1, p2)
                                       for p1 in A_copy for p2 in B_copy]

            def are_connected(p1, p2):
                return 1 if p1 in p2.connected_points else 0

            a_, b_ = max(all_dvalue_combinations,
                         key=lambda x: x[0].Dvalue + x[1].Dvalue - 2 * are_connected(x[0], x[1]))

            gain += a_.Dvalue + b_.Dvalue - 2 * are_connected(a_, b_)

            #remove them from the sets
            A_copy.remove(a_)
            B_copy.remove(b_)

            x_star.append(a_)
            y_star.append(b_)

            #buffer the current gain
            gain_buffer.append({
                'k': k,
                'gain': gain,
                'x_star': x_star.copy(),
                'y_star': y_star.copy()
            })

            #update D values for A and B 
            for point in A_copy:
                point.Dvalue = point.Dvalue + 2 * \
                    are_connected(a_, point) - 2*are_connected(b_, point)

            for point in B_copy:
                point.Dvalue = point.Dvalue + 2 * \
                    are_connected(b_, point) - 2*are_connected(a_, point)

        max_gain = max(gain_buffer, key=lambda x: x['gain'])
        #if the best gain is <= 0, stop
        if int(max_gain['gain']) <= 0:
            break
        else:
            #something has changed here
            indicator +=1
            
            # interchange points
            for a_, b_ in zip(max_gain['x_star'], max_gain['y_star']):
                print([i.index for i in max_gain['x_star']])
                A.append_outer_point(b_)
                B.append_outer_point(a_)

                A.remove_outer_point(a_)
                B.remove_outer_point(b_)

                groups_[a_.index] = B.name
                groups_[b_.index] = A.name

    return indicator
