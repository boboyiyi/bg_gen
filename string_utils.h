#ifndef STRING_UTILS_H
#define STRING_UTILS_H
#include <cctype>
#include <functional>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
# define strtok_r strtok_s
#endif

std::string
trim_left(std::string &s){
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

std::string
trim_right(std::string &s){
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

std::string
tiny_trim(std::string &s){
    return trim_left(trim_right(s));
}

std::vector<std::string>
tiny_split(std::string &str, const char *delimiters) {
    std::string sentry = str;
    std::vector<std::string> tokens;
    char *saveptr;
    char *token;
    for(token = strtok_r((char *)sentry.c_str(), delimiters, &saveptr);
        token != NULL;
        token = strtok_r(NULL, delimiters, &saveptr)) {
        tokens.push_back(std::string(token));
    }
    return tokens;
}

#endif