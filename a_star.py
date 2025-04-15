import networkx as nx
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function

def astar_path(G, source, target, visited_nodes=None, heuristic=None, weight="weight", cutoff=None, backtrack_limit=0):
    """
    A* pathfinding with a backtracking limit - Adaptation from networkX.

    Parameters:
    - G: NetworkX graph
    - source: Starting node
    - target: Goal node
    - visited_nodes: Set of nodes to limit backtracking (default: empty set)
    - heuristic: Heuristic function h(u, v) (default: 0)
    - weight: Edge weight attribute or function (default: "weight")
    - cutoff: Maximum f-cost to explore (default: None)
    - backtrack_limit: Maximum number of backtracks allowed (default: 0)

    Returns:
    - path: List of nodes from source to target
    - backtracks: Number of nodes in the path that are in visited_nodes

    Raises:
    - nx.NetworkXNoPath: If no path exists within the backtrack limit
    """
    visited_nodes = set(visited_nodes) if visited_nodes is not None else set()
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Either source {source} or target {target} is not in G")

    if heuristic is None:
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)
    G_succ = G._adj

    c = count()
    # Queue items: (f-cost, counter, node, g-cost, backtracks, parent_state)
    queue = [(0, next(c), source, 0, 0, None)]
    # enqueued: (node, backtracks) -> (g-cost, h-value)
    enqueued = {}
    # explored: (node, backtracks) -> (parent_node, parent_backtracks)
    explored = {}

    while queue:
        _, __, curnode, dist, b, parent_state = pop(queue)

        if curnode == target:
            path = [curnode]
            state = parent_state
            while state is not None:
                parent_node, _ = state
                path.append(parent_node)
                state = explored[state]
            path.reverse()
            return path, b, dist

        state = (curnode, b)
        if state in explored:
            if explored[state] is None:
                continue
            qcost, h = enqueued[state]
            if qcost < dist:
                continue

        explored[state] = parent_state

        for neighbor, w in G_succ[curnode].items():
            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            # Increment backtracks if neighbor is in visited_nodes
            b_neighbor = b + 1 if neighbor in visited_nodes else b
            if b_neighbor > backtrack_limit:
                continue

            ncost = dist + cost
            neighbor_state = (neighbor, b_neighbor)
            if neighbor_state in enqueued:
                qcost, h = enqueued[neighbor_state]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)

            if cutoff and ncost + h > cutoff:
                continue

            enqueued[neighbor_state] = (ncost, h)
            push(queue, (ncost + h, next(c), neighbor, ncost, b_neighbor, state))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} with backtrack_limit={backtrack_limit}")



def astar_path_with_backtracking(G, source, target, visited_nodes=None, heuristic=None, weight="weight", cutoff=None,
                                 backtrack_mode='iterative', percentage=None, max_backtracks=100, target_distance=None, lengths=None):
    if backtrack_mode == 'iterative': # Refactoring this to not include lengths, target_dis and the extra astar_path, does improve performance. 
        for k in range(max_backtracks + 1):
            try:
                path, b, dist = astar_path(G, source, target, visited_nodes, heuristic, weight, cutoff, backtrack_limit=k)
                if lengths is not None:
                    manual_distance = sum(lengths[path[i]][path[i + 1]] for i in range(len(path) - 1))
                else:
                    manual_distance = sum(G.edges[path[i], path[i + 1], 0]['length'] for i in range(len(path) - 1))
                if target_distance is None:
                    return path, b
                if manual_distance <= (1000*target_distance):
                    return path, b
                continue
            except nx.NetworkXNoPath:
                continue
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} with backtrack_limit={max_backtracks}")

    elif backtrack_mode == 'percentage':
        if percentage is None or not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        try:
            shortest_path = nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)
            L_estimate = len(shortest_path)
            k = int(percentage * L_estimate)
            path, b, dist = astar_path(G, source, target, visited_nodes, heuristic, weight, cutoff, backtrack_limit=k)
            if b <= int(percentage * len(path)):
                return path, b
            else:
                raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} with percentage={b}%")
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} with percentage={percentage*100}%")

    elif backtrack_mode == 'limit':
        try:
            path, b, dist = astar_path(G, source, target, visited_nodes, heuristic, weight, cutoff, backtrack_limit=max_backtracks)
            return path, b
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(f"Node {target} not reachable from {source} with percentage={max_backtracks}")
        else:
            raise ValueError("For 'limit' mode, max_backtracks should be an integer backtrack limit")
