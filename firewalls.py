#---- firewalls related

def find_firewalls(array_of_subsets, protect=None):
    if protect == 'self' or protect == None:
        return self_defense_variation(array_of_subsets)
        # ------------------------------- this is another variation that protects the connected nodes
    elif protect == 'connected':
        return connceted_defense_variation_noEdges(array_of_subsets)

def find_potential_firewalls(array_of_subsets):
    arrayOfEdgePoints = []
    for group in array_of_subsets:
        if group.outerPoints:
            arrayOfEdgePoints = arrayOfEdgePoints + list(group.outerPoints)

    potentialFirewalls = []
    if arrayOfEdgePoints:
        for point in arrayOfEdgePoints:
            outerEdges = [
                p for p in point.connected_points if p.group != point.group]
            potentialFirewalls.append(
                {'point': point, 'outerEdges': outerEdges})

    return potentialFirewalls

def self_defense_variation(array_of_subsets):
    actualFirewalls = []
    protectedEdges = []
    # This finds all the potential ( outer points ) firewalls, then sort them based NUMBER of outer edges
    potentialFirewalls = sorted(find_potential_firewalls(
        array_of_subsets), key=lambda x: len(x['outerEdges']), reverse=True)

    if potentialFirewalls:  # if there are outerpoints:
        """
        The following code is to remove the the current selected candidate from the count of other potentials
        outer edge, why?
        because once we assign a firewall, all its edges will be protected, and hence, must be removed
        from the count of the following selections.
        """
        for potential in potentialFirewalls:  # this segemnts is to remove the
            for potential2 in potentialFirewalls:
                if potential['point'] in potential2['outerEdges']:
                    potential2['outerEdges'].remove(potential['point'])
            # if the potential still has edges the above removal of already-protected edges
            if(potential['outerEdges']):
                current_selected_firewall = potential['point']
                current_selected_firewall.state = 1
                actualFirewalls.append(current_selected_firewall)

                # this assignes the value 2 which means protected to the points connected to the current firewall
                for point in current_selected_firewall.connected_points:
                    protectedEdges.append(
                        [(current_selected_firewall.x, current_selected_firewall.y), (point.x, point.y)])

            # Sort the current potientials based on number of outer edges again
            potentialFirewalls = sorted(
                potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)
    return actualFirewalls, protectedEdges

def connceted_defense_variation(array_of_subsets):
    actualFirewalls = []
    protectedEdges = []
    # This finds all the potential ( outer points ) firewalls, then sort them based NUMBER of outer edges
    potentialFirewalls = sorted(find_potential_firewalls(
        array_of_subsets), key=lambda x: len(x['outerEdges']), reverse=True)

    if potentialFirewalls:  # if there are outerpoints:
        """
        The following code is to remove the the current selected candidate from the count of other potentials
        outer edge, why?
        because once we assign a firewall, all its edges will be protected, and hence, must be removed
        from the count of the following selections.
        """
        for potential in potentialFirewalls:
            for potential2 in potentialFirewalls:
                if potential['point'] in potential2['outerEdges']:
                    potential2['outerEdges'].remove(potential['point'])
            # if the potential still has edges the above removal of already-protected edges
            if(potential['outerEdges']):
                current_selected_firewall = potential['point']
                if current_selected_firewall.state != 2:
                    current_selected_firewall.state = 1
                    actualFirewalls.append(current_selected_firewall)

                # this assignes the value 2 which means protected to the points connected to the current firewall
                for point in current_selected_firewall.connected_points:
                    """
                    For future reference: if applying a simulation about infection, we will need also to update the edge matrix tha ease the process of tracking protected edges.
                    """
                    point.state = 2
                    protectedEdges.append(
                        [(current_selected_firewall.x, current_selected_firewall.y), (point.x, point.y)])
                    # now we also protect the edges of the protected (state: 2)  points
                    for point2 in point.connected_points:
                        protectedEdges.append(
                            [(point.x, point.y), (point2.x, point2.y)])

            # <-- keeps only the unprotected potential candidates
            potentialFirewalls = [
                firewall for firewall in potentialFirewalls if firewall['point'].state != 2]

            # Sort the current potientials based on number of outer edges again            
            potentialFirewalls = sorted(
                potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)

    return actualFirewalls, protectedEdges

def connceted_defense_variation_noEdges(array_of_subsets):
    actualFirewalls = []
    protected_points = []
    # This finds all the potential ( outer points ) firewalls, then sort them based NUMBER of outer edges
    potentialFirewalls = sorted(find_potential_firewalls(
        array_of_subsets), key=lambda x: len(x['outerEdges']), reverse=True)

    if potentialFirewalls:  # if there are outerpoints:
        """
        The following code is to remove the the current selected candidate from the count of other potentials
        outer edge, why?
        because once we assign a firewall, all its edges will be protected, and hence, must be removed
        from the count of the following selections.
        """
        for potential in potentialFirewalls:
            for potential2 in potentialFirewalls:
                if potential['point'] in potential2['outerEdges']:
                    potential2['outerEdges'].remove(potential['point'])
            # if the potential still has edges the above removal of already-protected edges
            if(potential['outerEdges']):
                current_selected_firewall = potential['point']
                if current_selected_firewall.state != 2:
                    current_selected_firewall.state = 1
                    actualFirewalls.append(current_selected_firewall)

                # this assignes the value 2 which means protected to the points connected to the current firewall
                for point in current_selected_firewall.connected_points:
                    """
                    For future reference: if applying a simulation about infection, we will need also to update the edge matrix tha ease the process of tracking protected edges.
                    """
                    point.state = 2
                    protected_points.append(point)
                    # now we also protect the edges of the protected (state: 2)  points

            # <-- keeps only the unprotected potential candidates
            potentialFirewalls = [
                firewall for firewall in potentialFirewalls if firewall['point'].state != 2]

            # Sort the current potientials based on number of outer edges again
            potentialFirewalls = sorted(
                potentialFirewalls, key=lambda x: len(x['outerEdges']), reverse=True)

    return actualFirewalls, protected_points


"""
should i put gain and x_star in while True ?

"""