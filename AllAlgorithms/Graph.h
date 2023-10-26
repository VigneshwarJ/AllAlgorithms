#pragma once
#include <vector>
#include <optional>
#include <stack>
// templates should be in a single file.
template <typename T> struct UnWeightedNode {
	int mId;
	T mValue;
	std::vector<int> mNeighbors;
	UnWeightedNode():mId(-1)
	{

	}
	UnWeightedNode(int id, T value, std::vector<int>& neighbors): mId(id),
		mValue(value), mNeighbors(std::move(neighbors)){}

	bool equals(T value) {
		return std::equal_to<>{}(mValue, value);
	}
};

template <typename T> struct WeightedNode {
	int mId;
	T mValue;
	std::vector<std::pair<int,int>> mNeighbors;
	WeightedNode() :mId(-1)
	{

	}
	WeightedNode(int id, T value, std::vector<std::pair<int, int>>& neighbors) : mId(id),
		mValue(value), mNeighbors(std::move(neighbors)) {}

	bool equals(T value) {
		return std::equal_to<>{}(mValue, value);
	}
};

// C++ templates class methods alone cannot be partially specialized
// remedy is to create a partially specialized class : Using inheritence to avoid redefinition.

template <typename T, typename Node> class GraphBase {
	
protected:
	std::vector<Node> mNodes;
	int depthFirstSearchRecursive(int root, T val, bool* visited);
	int depthFirstSearchIter(int root, T val, bool* visited);
	int breadthFirstSearchIter(int root, T val, bool* visited);
public:
	static int Count;
	bool depthFirstSearch(T val, bool recursive = true);
};

template <typename T,typename Node = UnWeightedNode<T>> class Graph : public GraphBase<T,Node>
{
	//c++ does not superclass templates for name resolution. //method 1
	using GraphBase<T, Node>::mNodes; //using Superclass<T>::g;
	using GraphBase<T, Node>::Count;//using Superclass<T>::b;
	
public:
	Graph();
	void addNode(T value, std::vector<int> neighbors);
	
};


template <typename T> class Graph<T, WeightedNode<T>> : public GraphBase<T, WeightedNode<T>>
{
	//c++ does not superclass templates for name resolution. // method 2 using this..
public:
	Graph()
	{

	}
	void addNode(T value, std::vector < std::pair<int, int>> neighbors)
	{
		this->mNodes.push_back(WeightedNode<T>(this->Count++, value, neighbors));
	}

};




template <typename T, typename Node>
int GraphBase<T,Node>::Count= 0;




template <typename T, typename Node>
Graph<T,Node>::Graph()
{

}

template <typename T, typename Node>
void Graph<T,Node>::addNode(T value, std::vector<int> neighbors)
{
	mNodes.push_back(Node(Count++, value, neighbors));
}



template <typename T, typename Node>
int GraphBase<T,Node>::depthFirstSearchRecursive(int root, T val, bool* visited)
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
			auto result = depthFirstSearchRecursive(value, val, visited);
			if (result)
			{
				return result;
			}
		}
		return -1;
	}
}

template <typename T, typename Node>
int GraphBase<T,Node>::depthFirstSearchIter(int root, T val, bool* visited)
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
			if(!visited[value])
				depthStack.push(value);
		}
	}
	return -1;
}

template <typename T, typename Node>
int GraphBase<T,Node>::breadthFirstSearchIter(int root, T val, bool* visited)
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

template <typename T, typename Node>
bool GraphBase<T,Node>::depthFirstSearch(T val, bool recursive)
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
