// AllAlgorithms.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include "Graph.h"
#include <set>
#include "ParallelExamples.h"
#include<chrono>
#include <cublas.h>

template <typename Func, typename ...Args>
auto printTime(Func func, Args&&... args)
{
    using FuncReturnType = decltype(func(std::forward<Args>(args)...));

    const auto start = std::chrono::high_resolution_clock::now();
    if constexpr (std::is_same_v<void, FuncReturnType>) {
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }
    else {
        FuncReturnType ret = func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        return std::tuple<FuncReturnType, double>{
            ret, std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
        };
    }
}
void printMat(float* mat,int m,int n)
{
    if (m > 25)
        return;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            std::cout << mat[i * m + j] <<" ";
        }
        std::cout << "\n";
    }
}

void checkMatEqual(float* mat, float* mat2, int m, int n)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            if (mat[i * m + j]!= mat2[i * m + j])
            {
                std::cout << i <<" "<<j<<  " "<< mat[i * m + j] << " " << mat2[i * m + j] << " fail\n";
                return;
            } 
        }
        
    }
}

void fillRand(float* mat, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*m+j] = rand() % 3 ;
        }
    }
}

struct vec4
{
    float a, b, c, d;
    vec4(float a, float b, float c, float d)
        :a(a),
        b(b),
        c(c),
        d(d){}
};
void test(int N)
{
    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);

    // ...

    /*auto start = std::chrono::high_resolution_clock::now();
    addVectorSISD(A.data(), B.data(), C.data(), N);
    auto end = std::chrono::high_resolution_clock::now();

    auto delta_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();*/
    //std::cout << printTime(addVectorSISD, A.data(), B.data(), C.data(), N) << std::endl;
    //std::cout << printTime(addVectorSIMT,A.data(), B.data(), C.data(), N) << std::endl;
#define N 1024
    float* AMat = new float[N*N];
    float* BMat = new float[N * N];
    fillRand((float*)AMat, N, N);
    fillRand((float*)BMat, N, N);
    float* CMat = new float[N * N];
    float* DMat = new float[N * N];

    
    std::cout << printTime(matrixMultiplySISD, (float*)AMat, (float*)BMat, 
        (float*)CMat, N, N, N)
        << std::endl;

    printMat((float*)CMat, N, N);
    std::cout << printTime(matrixMultiplySIMTNaive, (float*)AMat, (float*)BMat, 
        (float*)DMat, N,N,N) << std::endl;
    checkMatEqual((float*)CMat, (float*)DMat, N, N);
    printMat((float*)DMat,N,N);

     

}

void GraphCheck()
{
    Graph<char> graph;
    graph.addNode('0', { 1,2 });
    graph.addNode('h', { 2 });
    graph.addNode('t', { 0,3 });
    graph.addNode('u', { 3 });
    bool result = graph.depthFirstSearch('u');
    std::cout << "Hello World!\n" << std::boolalpha << result << "\n";
    Graph<std::string> graph2;
    graph2.addNode("00", { 1,2 });
    graph2.addNode("hello", { 2 });
    graph2.addNode("test", { 0,3 });
    graph2.addNode("u", { 3 });
    result = graph2.depthFirstSearch("u", false);
    std::cout << "Hello World!\n" << std::boolalpha << result << "\n";

    Graph<char, WeightedEdge<char>> graph3;
    graph3.addNode('0', { {1,2},{2,3} });
    graph3.addNode('h', { {2 ,2} });
    graph3.addNode('t', { {0,3},{3,2} });
    graph3.addNode('u', { {3,2} });
    //result = graph3.depthFirstSearch('h');
    result = graph3.breadthFirstSearch('u');
    std::cout << "Hello World!\n" << std::boolalpha << result << "\n";
}

int main()
{
    test(10000000000000);
}

