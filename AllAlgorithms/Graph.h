#pragma once
#include <vector>
#include <optional>
#include <stack>
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
	int breadthFirstSearchIter(int root, T val, bool* visited);
public:
	static int Count;
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
		if (mNodes[top].equals(val))
		{
			return top;
		}
		visited[top] = true;
		for (auto& value : mNodes[top].mNeighbors)
		{
			if(!visited[value.toEdge])
				depthStack.push(value.toEdge);
		}
	}
	return -1;
}

template <typename T, typename E>
int Graph<T,E>::breadthFirstSearchIter(int root, T val, bool* visited)
{
	//std::heap<int> depthStack;
	//depthStack.push(root);
	//while (!depthStack.empty())
	//{
	//	auto top = depthStack.top();
	//	depthStack.pop();
	//	if (Nodes[top].equals(val))
	//	{
	//		return top;
	//	}
	//	visited[top] = true;
	//	for (auto& value : Nodes[top].mNeighbors)
	//	{
	//		depthStack.push(value);
	//	}
	//}
	return -1;
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