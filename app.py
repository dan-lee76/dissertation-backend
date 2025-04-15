from flask import Flask, request, render_template
from flask_cors import CORS
from main import AStarArcs
from graph_builder import Graph_Builder

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/route', methods=['GET'])
def get_route():
    distance = int(request.args.get('distance'))
    print(request.args)
    start_coords = request.args.get('start').split(',')
    end_coords = request.args.get('end').split(',')
    peak_bagger = request.args.get('peak_bagger')
    tolerance = int(request.args.get('tolerance'))
    print(distance, start_coords, end_coords)
    graph_builder.set_start_node(start_coords)
    graph_builder.set_end_node(end_coords)

    astar = AStarArcs(graph_builder, distance, tolerance)
    astar.boundary_generation()
    if peak_bagger == 'true':
        routes = astar.peak_bagger_generation()
        path = astar.get_route(routes, best_fitness=True, most_peaks=True)
    else:
        routes = astar.normal_generation()
        path = astar.get_route(routes, best_fitness=True, most_peaks=True)
    if path == None:
        return "No path found"

    return path, 10


graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822),simplify=False, server=True)
if __name__ == '__main__':
    app.run()