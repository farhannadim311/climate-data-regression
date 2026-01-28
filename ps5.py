# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re
import warnings

warnings.simplefilter('ignore', pylab.RankWarning)
# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    res = []
    for d in degs:
        res.append(pylab.polyfit(x,y,d))
    return res
def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    num = (y - estimated)
    num = num**2
    denom = (y - y.mean())
    denom = denom**2
    n = 0
    d = 0
    for val in num:
        n = n + val
    for val in denom:
        d = d + val
    return 1 - (n/d)




def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    
    for m in models:
        title = ""
        pylab.scatter(x,y, color = 'blue', marker = 'o')
        pylab.xlabel("Years")
        pylab.ylabel("Degree Celsius")
        slope = 0.0
        reg = pylab.polyval(m, x)
        if(len(m) == 2):
            slope = se_over_slope(x,y,reg,m)
            title = title + "Standard Error over slope is" + " " + str(slope) 
        r = r_squared(y,reg)
        title = title + "R^2 is " + " "+  str(r)
        title = title + "Degree is "+  " " + str(len(m) - 1)
        title = title + "\n"
        pylab.title(title)
        pylab.plot(x,reg, 'r-')
        pylab.show() 








def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    res = []
    for y in years:
        total_year = []
        average_year = []
        for idx,city in enumerate(multi_cities):
            total_year.append(climate.get_yearly_temp(city, y))
            average_year.append(total_year[idx].mean())
        average_year = pylab.array(average_year)
        res.append(average_year.mean())
    res = pylab.array(res)
    return res

        


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    res = []
    for i in range(len(y)):
        idx = 0
        total = 0
        avg = 0
        if(i - (window_length - 1)> 0):
            for j in range((i - (window_length -1)), i + 1):
                total = total + y[j]
                idx += 1
        else:
            for k in range(i + 1):
                total = total + y[k]
                idx += 1
        avg = total / idx
        res.append(avg)
    res = pylab.array(res)
    return res



def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    i = len(y)
    tmp = (y - estimated)**2
    num = 0
    for n in tmp:
        num += n
    return (num/i)**0.5
#y = pylab.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
#estimate = pylab.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
#correct = 35.8515457593
#print(rmse(y,estimate))
def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    # TODO
    pass

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for m in models:
        title = ""
        pylab.scatter(x,y, color = 'blue', marker = 'o')
        pylab.xlabel("Years")
        pylab.ylabel("Degree Celsius")
        reg = pylab.polyval(m, x)
        r = rmse(y, reg)
        title = title + "RMSE is " + " "+  str(r)
        title = title + "Degree is "+  " " + str(len(m) - 1)
        title = title + "\n"
        pylab.title(title)
        pylab.plot(x,reg, 'r-')
        pylab.show() 


    

if __name__ == '__main__':


    # Part A.4
    #temp = []
    #c = Climate("data.csv")
    #for year in TRAINING_INTERVAL:
        #temp.append(c.get_daily_temp("NEW YORK", 1, 10, year))
    #x = pylab.array(TRAINING_INTERVAL)
    #y = pylab.array(temp)
    #model = generate_models(x,y, [1])
    #evaluate_models_on_training(x,y,model)
    #avg = []
    #x = pylab.array(TRAINING_INTERVAL)
    #for year in TRAINING_INTERVAL:
        #temp = c.get_yearly_temp("NEW YORK", year)
        #avg.append(temp.mean())
    #avg = pylab.array(avg)
    #model = generate_models(x,avg,[1])
    #evaluate_models_on_training(x,avg,model)
    
    # Part B
   #c = Climate("data.csv") 
   #degs = [1]
   #years = []
   #for i in TRAINING_INTERVAL:
       #years.append(i)
   #years = pylab.array(years)
   #y = gen_cities_avg(c, CITIES, TRAINING_INTERVAL)
   #model = generate_models(years,y, degs)
   #evaluate_models_on_training(years,y,model)


    # Part C
    #c = Climate("data.csv")
    #degs = [1]
    #years = []
    #for i in TRAINING_INTERVAL:
        #years.append(i)
    #years = pylab.array(years)
    #y = gen_cities_avg(c, CITIES, TRAINING_INTERVAL)
    #y = moving_average(y, 5)
    #model = generate_models(years, y, degs)
    #evaluate_models_on_training(years,y,model)
    # Part D.2
    

    # Part E
    # TODO: replace this line with your code
    