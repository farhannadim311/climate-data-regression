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

#c = Climate("data.csv")
#a = c.get_yearly_temp("SEATTLE", 1964)
#print(len(a))
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
    res = []
    for y in years:
        std_year = []
        daily_avg = []
        for m in range(1, 13):
            if(m == 1 or m == 3 or m == 5 or m == 7 or m == 8 or m == 10 or m == 12):
                for d in range(1,32):
                    temp = 0
                    for idx, city in enumerate(multi_cities):
                        temp += climate.get_daily_temp(city, m, d,y) #Jan 1st
                    daily_avg.append(temp / len(multi_cities))
            elif (m == 4 or m == 6 or m ==9 or m == 11):
                for d in range(1,31):
                    temp = 0
                    for idx, city in enumerate(multi_cities):
                        temp += climate.get_daily_temp(city,m,d,y)
                    daily_avg.append(temp / len(multi_cities))         
            else:
                if(y % 4 == 0 and y % 100 != 0) or (y % 400 == 0):
                    for d in range(1,30):
                        temp = 0
                        for idx, city in enumerate(multi_cities):
                            temp += climate.get_daily_temp(city,m,d,y)
                        daily_avg.append(temp / len(multi_cities))
                else:
                    for d in range(1,29):
                        temp = 0
                        for idx, city in enumerate(multi_cities):
                            temp += climate.get_daily_temp(city,m,d,y)
                        daily_avg.append(temp / len(multi_cities))

        daily_avg = pylab.array(daily_avg)
        avg = 0
        for city in multi_cities:
            total = climate.get_yearly_temp(city,y)
            avg += total.mean()
        avg = avg / len(multi_cities) 
        #print(daily_avg)
        for idx,city in enumerate(multi_cities):
            #print(total)
            var = (daily_avg - avg)**2
            #print(var)
            #print(len(total))
            std = (var.sum() / len(daily_avg))**0.5
            std_year.append(std)
        #print(len(std_year))
        std_year = pylab.array(std_year)
        res.append(std_year.mean())
    return res

#climate = Climate('data.csv')
#years = pylab.array(TRAINING_INTERVAL)
#result = gen_std_devs(climate, CITIES, years)
#correct = [6.1119325255476635, 5.4102625076401125, 6.0304210441394801, 5.5823239710637846, 5.5908151965372177, 5.0347634736031583, 6.2485081784971772, 5.6752637253518703, 5.9822493041266327, 5.5376216719090898, 6.0339331562285095, 6.3471434661632733, 5.3872564859222782, 5.7528361897357705, 6.0117329392620285, 5.5922579610955854, 5.67888175212234, 5.7810899373043272, 5.7184178577664087, 5.3955809402004036, 5.1736886920193665, 5.8134229790176573, 5.1915733214759872, 5.4023314139519591, 6.7868442109830855, 5.2952870947334114, 5.6064597624296333, 5.4921097908102086, 6.1450202825415214, 6.3591021848005278, 5.4996866353350615, 5.6516820894310058, 5.7969983303071411, 5.8531227958031931, 5.2545492072097808, 6.0102701017450126, 5.5327493838092865, 5.7703034605336532, 5.0412624972468443, 5.2728662938897264, 5.0859211734722649, 5.5526426823734987, 5.8005720594546748, 5.7391426965165389, 5.5518538235632207, 5.8279562142168073, 5.9089508390885479, 5.9789908401877394, 6.5696153940105573]
#print(result)
#correct = [6.8007729489975439, 6.9344723094071865, 7.2965004501815818, 6.8077243598168549, 6.5055948680511539, 6.959087494608867, 6.4889799240243695, 6.9510430337868963, 7.0585431115159478, 7.0977420580318782, 6.8386579785236048, 6.731347077523127, 6.6616225764762902, 6.4092396746786013, 6.6214217100011084, 6.7136104957814435, 7.2575482189983553, 7.263276360210706, 7.1787611973720633, 7.0859352578611796, 6.8736741252762821, 6.7957043866857889, 7.0815549177622765, 6.7249974778654433, 7.2162729580931124, 6.4560372283957266, 6.7288306794528907, 6.9720986945202927, 6.922958341746317, 6.3033645588306086, 6.5330170805999908, 6.2777429551963237, 6.8488629387504032, 6.8257830274740625, 6.7856101061465059, 6.7592782215870484, 6.6634050127541604, 6.4486321701001552, 6.3413248952817742, 6.7637674361128752, 6.5519930751275384, 6.6831654464946064, 6.7751550280705839, 6.7435411127318146, 6.8720508861149154, 6.381528250607194, 6.9707944558310109, 6.7582457290380731, 6.7451346848899991]
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
    c = Climate("data.csv")
    #for year in TRAINING_INTERVAL:
        #temp.append(c.get_daily_temp("NEW YORK", 1, 10, year))
    #x = pylab.array(TRAINING_INTERVAL)
    #y = pylab.array(temp)
    #model = generate_models(x,y, [1])
    #evaluate_models_on_training(x,y,model)
    avg = []
    x = pylab.array(TRAINING_INTERVAL)
    for year in TRAINING_INTERVAL:
        temp = c.get_yearly_temp("NEW YORK", year)
        avg.append(temp.mean())
    avg = pylab.array(avg)
    model = generate_models(x,avg,[1])
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
    #c = Climate("data.csv")
    #training_years = []
    #degs = [1,2,20]
    #for i in TRAINING_INTERVAL:
        #training_years.append(i)
    #training_years = pylab.array(training_years)
    #y = gen_cities_avg(c, CITIES,TRAINING_INTERVAL)
    #y = moving_average(y,5)
    #model = generate_models(training_years, y, degs)
    #evaluate_models_on_training(training_years, y, model)
    #years = []
    #for i in TESTING_INTERVAL:
        #years.append(i)
    #years = pylab.array(years)
    #y = gen_cities_avg(c, CITIES, TESTING_INTERVAL)
    #y = moving_average(y,5)
    #evaluate_models_on_testing(years,y,model)

    # Part E
    #: replace this line with your code
    c = Climate("data.csv")
    training_years = []
    degs = [1]
    for i in TRAINING_INTERVAL:
        training_years.append(i)
    training_years = pylab.array(training_years)
    y = gen_std_devs(c, CITIES,TRAINING_INTERVAL)
    y = moving_average(y,5)
    model = generate_models(training_years, y, degs)
    evaluate_models_on_training(training_years, y, model)
  
    
    