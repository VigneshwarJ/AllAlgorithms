// AllAlgorithms.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include "Graph.h"
#include <set>



int main()
{
    Graph<char> graph;
    graph.addNode('0', { 1,2 });
    graph.addNode('h', {2});
    graph.addNode('t', { 0,3 });
    graph.addNode('u', {3});
    bool result = graph.depthFirstSearch('t');
    std::cout << "Hello World!\n" << std::boolalpha << result <<"\n";
    Graph<std::string> graph2;
    graph2.addNode("00", { 1,2 });
    graph2.addNode("hello", { 2 });
    graph2.addNode("test", { 0,3 });
    graph2.addNode("u", { 3 });
    result = graph2.depthFirstSearch("00",false);
    std::cout << "Hello World!\n" << std::boolalpha << result << "\n";

    Graph<char,WeightedNode<char>> graph3;
    graph3.addNode('0', { {1,2},{2,3} });
    graph3.addNode('h', { {2 ,2} });
    graph3.addNode('t', { {0,3},{3,2} });
    graph3.addNode('u', { {3,2} });
    result = graph3.depthFirstSearch('0', false);
}

