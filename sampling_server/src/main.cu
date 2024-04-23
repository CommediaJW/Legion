#include "server.h"
#include <stdlib.h>
#include <iostream>
#include <string>

std::vector<std::string> split(std::string s, std::string delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

int main(int argc, char **argv)
{

    std::cout << "Start Sampling Server\n";
    Server *server = NewGPUServer();
    auto fanout_str = split(argv[3], ",");
    std::vector<int> fanout;
    for (int32_t i = fanout_str.size() - 1; i >= 0; i--)
    {
        fanout.push_back(atoi(fanout_str[i].c_str()));
    }
    server->Initialize(atoi(argv[1]), fanout, 1); // gpu number, default 1; in memory, default true
    server->PreSc(atoi(argv[2]));                 // cache aggregate mode, default 0
    server->Run();
    server->Finalize();
}