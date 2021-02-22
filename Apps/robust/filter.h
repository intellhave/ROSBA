//Filter implementation
#ifndef FILTER_H
#define FILTER_H

#include <iomanip>
#include <vector>

struct Filter
{
    Filter(double alpha = 0.01) : _alpha(alpha)
    {
    }

    //Check if the pair (f, h) belongs to the forbidden region
    bool isDominated(double const &f, double const &h)
    {
        // return false;
        for (int i = 0; i < _elements.size(); ++i)                        
        {
            std::pair<double, double> e = _elements[i];
            if (f >= e.first && h >= e.second)
                return true;
        }
        return false;
    }

    void setAlpha(double const &alpha){
        _alpha = alpha;
    }

    bool isAccepted (double const &f, double const &h){
        return !isDominated(f, h);
    }

    //Add one more element to the filter
    void addElement(double const &f, double const &h, double const &cost)
    {
        double alpha = _alpha;

        double fn = f - alpha * h;
        double hn = h - alpha * h;
        _elements.push_back(std::make_pair(fn, hn));
    }

    //Add one more element to the filter
    void addElement(double const &f, double const &h)
    {
        _elements.push_back(std::make_pair(f - _alpha * h, (1 - _alpha) * h));
    }

    //Remove last element
    void removeLastElement()
    {
        _elements.erase(_elements.end() - 1);
    }

    //Return the current filter size
    int getSize() { return _elements.size(); }

    private:
    double _alpha;
    std::vector<std::pair<double, double>> _elements;

}; //end struct filter

#endif
