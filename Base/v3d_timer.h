// -*- C++ -*-

#ifndef V3D_TIMER_H
#define V3D_TIMER_H

#include <cstdio>
#include <ctime>
#include <algorithm>
#include <ctime>

#if defined(WIN32) && !defined(__MINGW32__)
# include <windows.h>
#else
# include <cstring>
# include <sys/time.h>

# define V3D_USE_CLOCK_GETTIME
# if defined(V3D_USE_CLOCK_GETTIME)
#  define V3D_CLOCK_ID CLOCK_THREAD_CPUTIME_ID
//#  define V3D_CLOCK_ID CLOCK_CLOCK_MONOTONIC
# endif
#endif


namespace V3D
{
   using namespace std;

   class Timer
   {
      public:
         Timer(const char *name = "<unnamed timer>", int history_size = 0)
            : _total_time(0), _history(0), _history_index(0), _count(0)
         {
            _name[0] = '\0';

#if defined(WIN32) && !defined(__MINGW32__)
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            _freq = freq.QuadPart;
            if (name)
               strcpy_s(_name, sizeof(_name), name);
            else
               strcpy_s(_name, sizeof(_name), "<unnamed timer>");
#else
            //_freq = 1000000; // gettimeofday() return microseconds.
            _freq = 1000000000UL; // clock_gettime() return nanoseconds.
            if (name)
               strncpy(_name, name, sizeof(_name));
            else
               strncpy(_name, "<unnamed timer>", sizeof(_name));
#endif
            if (history_size > 0)
               _history = new unsigned long long[history_size];
            _history_size = history_size;
            std::fill(_history,_history+_history_size,0);
         }

         ~Timer()
         {
             delete[] _history;
         }

         void reset()
         {
            _total_time = _count = 0;
         }

         void start()
         {
#if defined(WIN32) && !defined(__MINGW32__)
            LARGE_INTEGER start_time;
            QueryPerformanceCounter(&start_time);
            _start_time = start_time.QuadPart;
#else
# if !defined(V3D_USE_CLOCK_GETTIME)
            timeval tv;
            gettimeofday(&tv, 0);
            _start_time =  tv.tv_sec*_freq + tv.tv_usec;
# else
            timespec ts;
            clock_gettime(V3D_CLOCK_ID, &ts);
            _start_time =  ts.tv_sec*_freq + ts.tv_nsec;
# endif
#endif
         }

         void stop()
         {
#if defined(WIN32) && !defined(__MINGW32__)
            LARGE_INTEGER cur_time;
            QueryPerformanceCounter(&cur_time);
            unsigned long long elapsed = cur_time.QuadPart - _start_time;
#else
# if !defined(V3D_USE_CLOCK_GETTIME)
            timeval tv;
            gettimeofday(&tv, 0);
            unsigned long long elapsed = tv.tv_sec*_freq + tv.tv_usec - _start_time;
# else
            timespec ts;
            clock_gettime(V3D_CLOCK_ID, &ts);
            unsigned long long elapsed = ts.tv_sec*_freq + ts.tv_nsec - _start_time;
# endif
#endif
            _total_time += elapsed;
            if (_history)
            {
               if (_count == _history_size)
                  _total_time -= _history[_history_index];
               else
                  ++_count;
               _history[_history_index] = elapsed;
               ++_history_index;
               if(_history_index >= _history_size)
                  _history_index = 0;
            }
            else
               ++_count;
         } // end stop()

         double getHertz() const { return _freq*(double)_count/_total_time; }
         double getTime()  const { return (double)_total_time/_freq; }

         unsigned long getCount() const { return _count; }
         const char *getName()    const { return _name; }

         void printHertz() const { printf("TIMING: %s: %.03f Hz\n",_name, this->getHertz()); fflush(stdout); }
         void printTime()  const { printf("TIMING: %s: %.03f s\n",_name, this->getTime()); fflush(stdout); }

         void print() const
         {
            printf("TIMING: %s: %.03f Hz, %.03f s/exec, %ld execs, %.03f s\n", _name,
                   getHertz(), getTime()/getCount(), getCount(), getTime());
            fflush(stdout);
         }

         void print(char const * info) const
         {
            printf("TIMING: %s (%s): %.03f Hz, %.03f s/exec, %ld execs, %.03f s\n", _name, info,
                   getHertz(), getTime()/getCount(), getCount(), getTime());
            fflush(stdout);
         }

      private:
         unsigned long long   _freq;
         unsigned long long   _total_time;
         unsigned long long * _history;
         int                  _history_size;
         int                  _history_index;
         unsigned long        _count;
         unsigned long long   _start_time;
         char                 _name[80];
   }; // end struct Timer


   struct ScopedTimer
   {
       ScopedTimer(const char *name = "<unnamed timer>") :
          _timer(name,0)
       {
          _timer.start();
       }

       ~ScopedTimer()
       {
          _timer.stop();
          _timer.print();
       }

   private:
       Timer _timer;
   };

} // end namespace V3D

#endif
