import networkx as nx
import numpy as np
from collections import defaultdict
from . import graphgen

def getHomoEdgeRatio(G:nx.Graph, **kwargs):
    homophily_count = 0
    total_edges = 0
    for u, v in G.edges():
        if G.node[v]['color'] > 0 and G.node[u]['color'] > 0:
            if G.node[v]['color'] == G.node[u]['color']:
                homophily_count += 1
            total_edges += 1
    homoEdgeRatio = float(homophily_count) / total_edges
    return {
        "homoEdgeRatio": homoEdgeRatio
    }

def getGeomGCNBeta(G:nx.Graph, **kwargs):
    nodeBeta = dict()
    for v in G.nodes():
        if G.node[v]['color'] > 0:
            degree = 0; betaCount = 0
            for u in G.neighbors(v):
                if G.node[u]['color'] > 0:
                    degree += 1
                    if G.node[v]['color'] == G.node[u]['color']:
                        betaCount += 1
            if degree > 0:
                nodeBeta[v] = betaCount / degree
    return {
        "GeomGCNBeta": sum(nodeBeta.values()) / len(nodeBeta)
    }

def getClassSize(ally, **kwargs):
    classSize = np.sum(ally, axis=0)
    return {
        "classSize": classSize
    }

def getDegrees(G:nx.Graph, **kwargs):
    degrees = list(map(len, G.adj.values()))
    avg_degree = np.mean(degrees)
    sorted_degree = np.array(sorted(degrees, reverse=True))
    return {
        "sorted_degree": sorted_degree,
        "avg_degree": avg_degree,
        "min_degree": min(sorted_degree),
        "max_degree": max(sorted_degree),
        "quantile_degree": np.quantile(sorted_degree, 
            [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    }

def getNumNodeEdges(G:nx.Graph, **kwargs):
    return {
        "numEdges": G.number_of_edges(),
        "numNodes": G.number_of_nodes()
    }

def getAvgCC(G:nx.Graph, **kwargs):
    return {
        "avgClusteringCoeff": nx.average_clustering(G)
    }

def getAvgShortestPath(G:nx.Graph, **kwargs):
    pair_count = 0; dist_count = 0
    for S in nx.connected_component_subgraphs(G):
        count = len(S.nodes) * (len(S.nodes) - 1)
        dist = nx.average_shortest_path_length(S) * count
        dist_count += dist; pair_count += count
    return {
        "avgSPLength": dist_count / pair_count
    }

def getNumComponents(G:nx.Graph, **kwargs):
    numComponents = len(list(nx.connected_components(G)))
    return {
        "numComponents": numComponents
    }

def getNumTriangles(G:nx.Graph, **kwargs):
    numTriangles = np.array(list(nx.triangles(G).values()))
    numTotalTriangles = np.sum(numTriangles) // 3
    return {
        "numTriangles": numTriangles,
        "numTotalTriangles": numTotalTriangles
    }

def getMatrixH(job, **kwargs):
    if job.sp.method == "mixhop":
        generator = graphgen.MixhopGraphGenerator(job.sp.classRatio, job.sp.heteroClsWeight, heteroWeightsExponent=job.sp.heteroWeightsExponent)
        return {
            "H": generator.getH(job.sp.h)
        } 
    else:
        return {
            "H": None
        }

def getEmpricialH(G:nx.Graph, ally, **kwargs):
    eH = np.zeros((ally.shape[1], ally.shape[1]))
    for (u, v) in G.edges:
        if G.node[v]['color'] > 0 and G.node[u]['color'] > 0:
            ul = G.nodes[u]['color'] - 1
            vl = G.nodes[v]['color'] - 1
            eH[ul, vl] += 1
            eH[vl, ul] += 1
    cH = eH
    eH = eH / eH.sum(1, keepdims=True)
    return {
        "cH": cH, 
        "eH": eH
    }


def getDataQuality(G:nx.Graph, ally, **kwargs):
    return {
        "numSelfLoops": G.number_of_selfloops(),
        "numNoLabel": np.sum(ally.sum(1) < 1)
    }

stats_dict = {
#  <stats_name>: (<stats_func>, <add_to_job_doc>, <add_to_job_data>)
#  Change <add_to_job_doc> or <add_to_job_data> to value larger than 1 can override previous value
    "homoEdgeRatio": (getHomoEdgeRatio, True, True),
    "classSize": (getClassSize, True, True),
    "sorted_degree": (getDegrees, False, True),
    "avg_degree": (getDegrees, True, True),
    "min_degree": (getDegrees, True, True),
    "max_degree": (getDegrees, True, True),
    "numEdges": (getNumNodeEdges, True, True),
    "numNodes": (getNumNodeEdges, True, True),
    "avgClusteringCoeff": (getAvgCC, True, True),
    "avgSPLength": (getAvgShortestPath, True, True),
    "numComponents": (getNumComponents, True, True),
    "numTriangles": (getNumTriangles, False, True),
    "numTotalTriangles": (getNumTriangles, True, True),
    "GeomGCNBeta": (getGeomGCNBeta, True, True),
    "H": (getMatrixH, False, True),
    "eH": (getEmpricialH, False, True),
    "cH": (getEmpricialH, False, True),
    "numSelfLoops": (getDataQuality, True, True),
    "numNoLabel": (getDataQuality, True, True),
    "quantile_degree": (getDegrees, True, True)
}
