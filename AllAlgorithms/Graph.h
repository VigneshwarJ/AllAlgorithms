#pragma once
#include <vector>
#include <queue>
#include <stack>
#include <iostream>
// templates should be in a single file.

struct Edge {
	int toEdge;
	Edge(int to):toEdge(to)
	{}
};
template<typename unit = int> struct WeightedEdge {
	int toEdge;
	unit weight;
};
template <typename T, typename E = Edge> struct Node {
	
	int mId;
	T mValue;
	std::vector<E> mNeighbors;
	Node():mId(-1)
	{

	}
	Node(int id, T value, std::vector<E>& neighbors): mId(id),
		mValue(value), mNeighbors(std::move(neighbors)){}

	bool equals(T value) {
		return std::equal_to<>{}(mValue, value); // using std::equal_to for comparision ( need to add compare template..
	}
};


template <typename T, typename E = Edge> class Graph {
	
protected:
	std::vector<Node<T,E>> mNodes;
	int depthFirstSearchRecursive(int root, T val, bool* visited);
	int depthFirstSearchIter(int root, T val, bool* visited);
public:
	static int Count;
	bool breadthFirstSearch( T val, int root = 0);
	bool depthFirstSearch(T val, bool recursive = true);
	Graph() {}
	void addNode(T value, std::vector<E> neighbors);
};




template <typename T, typename E>
int Graph<T,E>::Count= 0;


template <typename T, typename E>
void Graph<T,E>::addNode(T value, std::vector<E> neighbors)
{
	mNodes.push_back(Node<T,E>(Count++, value, neighbors));
}



template <typename T, typename E>
int Graph<T,E>::depthFirstSearchRecursive(int root, T val, bool* visited)
{
	if (visited[root])
	{
		return -1;
	}
	visited[root] = true;
	if (mNodes[root].equals(val))
	{
		return root;
	}
	else
	{
		for (auto& value : mNodes[root].mNeighbors)
		{
			auto result = depthFirstSearchRecursive(value.toEdge, val, visited);
			if (result != -1)
			{
				return result;
			}
		}
		return -1;
	}
}

template <typename T, typename E>
int Graph<T,E>::depthFirstSearchIter(int root, T val, bool* visited)
{
	std::stack<int> depthStack;
	depthStack.push(root);
	while (!depthStack.empty())
	{
		auto top = depthStack.top();
		depthStack.pop();
		visited[top] = true; 
		std::cout << top << " --> ";
		if (mNodes[top].equals(val))
		{
			std::cout << std::endl;
			return top;
		}
		for (auto& value : mNodes[top].mNeighbors)
		{
			if(!visited[value.toEdge])
				depthStack.push(value.toEdge);
		}
	}
	std::cout << std::endl;
	return -1;
}

template <typename T, typename E>
bool Graph<T,E>::breadthFirstSearch(T val, int root )
{
	bool* visited = new bool[mNodes.size()] {false};
	std::queue<int> breadthQueue;
	breadthQueue.push(root);
	while (!breadthQueue.empty())
	{
		auto front = breadthQueue.front();
		breadthQueue.pop();
		visited[front] = true;
		std::cout << front << " --> ";
		if (mNodes[front].equals(val))
		{
			std::cout << std::endl;
			return true;
		}
		for (auto& value : mNodes[front].mNeighbors)
		{
			if (visited[value.toEdge])
			{
				continue;
			}
			breadthQueue.push(value.toEdge);
		}
	}
	std::cout << std::endl;
	return false;
}

template <typename T, typename E>
bool Graph<T,E>::depthFirstSearch(T val, bool recursive)
{
	bool* visited = new bool[mNodes.size()] {false};
	int returned = -1;
	if (recursive)
	{
		returned = depthFirstSearchRecursive(0, val, visited);
	}
	else
		returned = depthFirstSearchIter(0, val, visited);
	delete[] visited;
	if (returned==-1)
		return false;
	return true;
}


/*
  Can a class method be partially specialized ? No ..
  C++ templates class methods alone cannot be partially specialized.
  Remedy is to create a partially specialized class : Using inheritence to avoid redefinition.
*/

/*
Can template be constrained ? No ...
https://stackoverflow.com/questions/874298/how-do-you-constrain-a-template-to-only-accept-certain-types
*/ 

/*
	//c++ does not superclass templates for name resolution
		- using Superclass<T>::g;
		- this-> ( to specify)
*/