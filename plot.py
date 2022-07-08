import numpy as np
from research_module import *
import plotly.express as px

def plot_points(ax, x, y, groups_, array_of_groups):

    ax.scatter(x, y, c=groups_, cmap='rainbow')

    # to plot edges
    for group in array_of_groups:
        for point1 in group.points:
            x1, y1 = (point1.x, point1.y)
            for point2 in point1.connected_points:
                x2, y2 = (point2.x, point2.y)
                ax.plot((x1, x2), (y1, y2), 'black')

    return ax

def plot_firewalls(ax, array_of_groups, protect):


    if protect == 'self':
        firewalls, edges = find_firewalls(array_of_groups, protect)

        main_firewall = []
        for point in firewalls:
            main_firewall.append([point.x, point.y])

        X = np.array(main_firewall)
        if len(X) != 0:
            ax.scatter(X[:, 0], X[:, 1], c='green')

            # plot the protected edges+
        for edge in edges:
            point_x = [edge[0][0], edge[1][0]]
            point_y = [edge[0][1], edge[1][1]]
            ax.plot(point_x, point_y, 'green')

    elif protect == 'connected':
        firewalls, protected_points = find_firewalls(array_of_groups, protect)

        main_firewall = []
        for point in firewalls:
            main_firewall.append([point.x, point.y])

        protected = []
        for point in protected_points:
            protected.append([point.x, point.y])

        X = np.array(main_firewall)
        if len(X) != 0:
            ax.scatter(X[:, 0], X[:, 1], c='green')

        X = np.array(protected)
        if len(X) != 0:
            ax.scatter(X[:, 0], X[:, 1], c='orange')

    return ax

def plot_borders(ax, xMax, xMin, yMax, yMin):

    ax.plot([(xMax+xMin)/4, (yMax+yMin)/4],
            [(xMax+xMin)/4, (yMax+yMin)/1.25], color='black')
    ax.plot([(xMax+xMin)/4, (yMax+yMin)/1.25],
            [(xMax+xMin)/1.25, (yMax+yMin)/1.25], color='black')
    ax.plot([(xMax+xMin)/1.25, (yMax+yMin)/1.25],
            [(xMax+xMin)/1.25, (yMax+yMin)/4], color='black')
    ax.plot([(xMax+xMin)/1.25, (yMax+yMin)/4],
            [(xMax+xMin)/4, (yMax+yMin)/4], color='black')

    return ax
