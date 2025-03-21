from flask import Flask, request
from flask_cors import CORS
from main import AStarArcs
from graph_builder import Graph_Builder

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/route', methods=['GET'])
def get_route():
    distance = int(request.args.get('distance'))
    start_coords = (float(request.args.get('start_lat')), float(request.args.get('start_lon')))
    end_coords = (float(request.args.get('end_lat')), float(request.args.get('end_lon')))
    astar = AStarArcs(graph_builder, distance, 1)
    astar.main()
    path = astar.ridge_walker()
    return path


graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822),simplify=True, server=True)
if __name__ == '__main__':
    app.run()