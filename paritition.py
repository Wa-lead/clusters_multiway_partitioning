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

        #the following if-statement will fail if both groups have nothing to interchange
        if gain_buffer:
            max_gain = max(gain_buffer, key=lambda x: x['gain'])
            #if the best gain is <= 0, stop
            if int(max_gain['gain']) <= 0:
                break
            else:
                #something has changed here
                indicator +=1
                
                # interchange points
                for a_, b_ in zip(max_gain['x_star'], max_gain['y_star']):
                    A.append_outer_point(b_)
                    B.append_outer_point(a_)

                    A.remove_outer_point(a_)
                    B.remove_outer_point(b_)

                    groups_[a_.index] = B.name
                    groups_[b_.index] = A.name

    return indicator


def cluster_size(point):
    seen_points = set()
    cluster_content = []
    def dfs(point):
        for p in point.connected_points:
            if p in seen_points or p.group != point.group:
                continue
            seen_points.add(p)
            cluster_content.append(p)
            dfs(p)

    dfs(point)
    return len(cluster_content)

def selection_hueristic1(current_candidates,threshold):   

    def hueristic(point):
        size = cluster_size(point) 
        if size > threshold:
            return float('-inf')
        else:
            return point.Dvalue

    return max(current_candidates, key= hueristic)

def single_way(A, B, groups_, def_size, margin = 0.1):
    indicator = 0

    UPPER_BOUND = (1 + margin) * def_size
    LOWER_BOUND = (1 - margin) * def_size

    # graph: if G > 0 redo the process
    A_copy = deepcopy(A)
    B_copy = deepcopy(B)
    
    gain_buffer = []
    gain = 0
    A_to_B = []

    THRESHOLD = min(len(A.points) - LOWER_BOUND, UPPER_BOUND - len(B.points))
    counter = 0
    
    #make sure that group A ( the one giving points ) still has outerpoints to give
    while counter <= THRESHOLD and A_copy.outerPoints:

        #compute D values for A and B (block1)
        for p1 in A_copy.outerPoints:
            internal_cost = sum(
                [1 if p2.group == A.name else 0 for p2 in p1.connected_points])
            external_cost = sum(
                [1 if p2.group == B.name else 0 for p2 in p1.connected_points])
            p1.Dvalue = external_cost - internal_cost

        #selection 
        cd_pt = max(A_copy.outerPoints, key=lambda x: x.Dvalue)
        # cd_pt = selection_hueristic1(A_copy.outerPoints, THRESHOLD - counter)

        gain+= cd_pt.Dvalue
        A_to_B.append(cd_pt)

        #transfer the point
        B_copy.append_outer_point(cd_pt)
        A_copy.remove_outer_point(cd_pt)

        #this updates if group A still connected to B
        B_copy.find_connected_groups()
        A_copy.find_connected_groups()

        gain_buffer.append({
            'k': counter,
            'gain': gain,
            'A_to_B': A_to_B.copy()
        })
        counter += 1

    #the following if-statement fails if one of the groups as the boundary
    if gain_buffer:
        max_gain = max(gain_buffer, key=lambda x: x['gain'])
        #if the best gain is <= 0, stop
        if int(max_gain['gain']) <= 0:
            pass
        else:
            #something has changed here
            indicator +=1
            
            #since the points are deepcopies of the original ones, we must prepare them first
            max_gain['A_to_B'] = [point for i in max_gain['A_to_B'] for point in A.outerPoints if point.index == i.index]

            # interchange points
            for a_ in max_gain['A_to_B']:
                B.append_outer_point(a_)
                A.remove_outer_point(a_)
                groups_[a_.index] = B.name

    return indicator