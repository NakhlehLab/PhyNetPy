import matplotlib.pyplot as plt
import abc_tree as abct
import statistics

# initializing vars 
N = 20 # number of different true rates from which trees will be simulated
div_pointx = [] # will hold the true diversification rates
div_pointy = [] # will hold the inferred diversification rates
turn_pointx = [] # will hold the true turnover rates
turn_pointy = [] # will hold the inferred turnover rates
births_pointx = [] # will hold the true birth shapes
births_pointy = [] # will hold the inferred birth shapes
deaths_pointx = [] # will hold the true death shapes
deaths_pointy = [] # will hold the inferred death shapes
subs_pointx = [] # will hold the true substitution shapes
subs_pointy = [] # will hold the inferred substitution shapes

i = 0 
while i < N:
    i += 1
    res_arr = abct.run_main(isreal_obs = False, num_accept = 5) # simulate a tree and infer the rates (from 10 accepted samples)
    
    # extract true rates from the returned array
    div_true = res_arr[6]
    turn_true = res_arr[7]
    births_true = res_arr[8]
    deaths_true = res_arr[9]
    subs_true = res_arr[10]

    # extract the accepted inferred rates from the returned array
    div_infer_arr = res_arr[0]
    turn_infer_arr = res_arr[1]
    births_infer_arr = res_arr[2]
    deaths_infer_arr = res_arr[3]
    subs_infer_arr = res_arr[4]

    # populating arrays with x (true rate) and y (mean inferred rate) coordinates
    div_pointx.append(div_true)
    div_pointy.append(statistics.mean(div_infer_arr))

    turn_pointx.append(turn_true)
    turn_pointy.append(statistics.mean(turn_infer_arr))

    births_pointx.append(births_true)
    births_pointy.append(statistics.mean(births_infer_arr))

    deaths_pointx.append(deaths_true)
    deaths_pointy.append(statistics.mean(deaths_infer_arr))

    subs_pointx.append(subs_true)
    subs_pointy.append(statistics.mean(subs_infer_arr))

def plot_div_exp_v_true():
    """
    Plotting mean inferred diversification rate vs true diversification rate.
    """
    plt.plot(div_pointx, div_pointy, 'ro')
    plt.xlabel('True diversification rate')
    plt.ylabel('Mean inferred diversification rate')
    plt.title('Mean inferred diversification rate vs true diversification rate')
    plt.show()

def plot_turn_exp_v_true():
    """
    Plotting mean inferred turnover rate vs true turnover rate.
    """
    plt.plot(turn_pointx, turn_pointy, 'ro')
    plt.xlabel('True turnover rate')
    plt.ylabel('Mean inferred turnover rate')
    plt.title('Mean inferred turnover rate vs true turnover rate')
    plt.show()

def plot_births_exp_v_true():
    """
    Plotting mean inferred birth shape vs true birth shape.
    """
    plt.plot(births_pointx, births_pointy, 'ro')
    plt.xlabel('True birth distribution shape')
    plt.ylabel('Mean inferred birth distribution shape')
    plt.title('Mean inferred birth distribution shape vs true birth distribution shape')
    plt.show()

def plot_deaths_exp_v_true():
    """
    Plotting mean inferred death shape vs true death shape.
    """
    plt.plot(deaths_pointx, deaths_pointy, 'ro')
    plt.xlabel('True death distribution shape')
    plt.ylabel('Mean inferred death distribution shape')
    plt.title('Mean inferred death distribution shape vs true death distribution shape')
    plt.show()

def plot_subs_exp_v_true():
    """
    Plotting mean inferred substitution shape vs true substitution shape.
    """
    plt.plot(subs_pointx, subs_pointy, 'ro')
    plt.xlabel('True substitution distribution shape')
    plt.ylabel('Mean inferred substitution distribution shape')
    plt.title('Mean inferred substitution distribution shape vs true substitution distribution shape')
    plt.show()

def calc_percent(true_arr, interval_arr):
    """
    Calculates the percent (as a fraction) that elements of 'true_arr'
    fall within the interval created by 'interval_arr[0]' and 'interval_arr[1]'
    (inclusive).
    """
    num_fails = 0 # counts number of times the rate is not in the interval
    for i in true_arr: # for each true rate
        if (i < interval_arr[0] or i > interval_arr[1]): # rate falls outside of the interval
            num_fails += 1
    return (len(true_arr) - num_fails) / len(true_arr) # equal to the number of rates in the interval / number of rates

def plot_coverage(infer_arr, true_arr, type):
    """
    Plotting fraction of elements in 'true_arr' (the true rates) that fall inside the 
    credible interval (created by the elements of 'infer_arr') vs the credible interval width.
    'type' specifies what rate/shape is being plotted.
    """
    total_interval = statistics.quantiles(infer_arr, n = 100) # splitting the inferred rates into 100 percentiles
    interval_50 = [] # will hold the middle 50% of inferred rates
    interval_55 = [] # will hold the middle 55% of inferred rates
    interval_60 = [] # will hold the middle 60% of inferred rates
    interval_65 = [] # will hold the middle 65% of inferred rates
    interval_70 = [] # will hold the middle 70% of inferred rates
    interval_75 = [] # will hold the middle 75% of inferred rates
    interval_80 = [] # will hold the middle 80% of inferred rates
    interval_85 = [] # will hold the middle 85% of inferred rates
    interval_90 = [] # will hold the middle 90% of inferred rates
    interval_95 = [] # will hold the middle 95% of inferred rates

    # creating credible intervals
    interval_50.append(total_interval[25])
    interval_50.append(total_interval[74])
    interval_55.append(total_interval[23])
    interval_55.append(total_interval[77])
    interval_60.append(total_interval[20])
    interval_60.append(total_interval[79])
    interval_65.append(total_interval[18])
    interval_65.append(total_interval[82])
    interval_70.append(total_interval[15])
    interval_70.append(total_interval[84])
    interval_75.append(total_interval[13])
    interval_75.append(total_interval[87])
    interval_80.append(total_interval[10])
    interval_80.append(total_interval[89])
    interval_85.append(total_interval[8])
    interval_85.append(total_interval[92])
    interval_90.append(total_interval[5])
    interval_90.append(total_interval[94])
    interval_95.append(total_interval[3])
    interval_95.append(total_interval[97])
   
    # calculating fraction of elements in 'true_arr' that fall inside 
    # credible intervals of varying sizes
    percent_arr = []
    percent_arr.append(calc_percent(true_arr, interval_50))
    percent_arr.append(calc_percent(true_arr, interval_55))
    percent_arr.append(calc_percent(true_arr, interval_60))
    percent_arr.append(calc_percent(true_arr, interval_65))
    percent_arr.append(calc_percent(true_arr, interval_70))
    percent_arr.append(calc_percent(true_arr, interval_75))
    percent_arr.append(calc_percent(true_arr, interval_80))
    percent_arr.append(calc_percent(true_arr, interval_85))
    percent_arr.append(calc_percent(true_arr, interval_90))
    percent_arr.append(calc_percent(true_arr, interval_95))

    plt.plot([50, 55, 60, 65, 70, 75, 80, 85, 90, 95], percent_arr, 'ro')
    plt.ylabel('Fraction of true ' + str(type) + ' that fall inside the credible interval')
    plt.xlabel('Credible interval width')
    plt.title('Coverage for ' + str(type))
    plt.show()

def plot_div_coverage(): 
    """
    Plotting fraction of true diversification rates that fall inside the
    credible interval vs the credible interval width.
    """
    plot_coverage(div_infer_arr, div_pointx, "diversification rates")

def plot_turn_coverage(): 
    """
    Plotting fraction of true turnover rates that fall inside the
    credible interval vs the credible interval width.
    """
    plot_coverage(turn_infer_arr, turn_pointx, "turnover rates")

def plot_births_coverage():
    """
    Plotting fraction of true birth shapes that fall inside the
    credible interval vs the credible interval width.
    """
    plot_coverage(births_infer_arr, births_pointx, "birth distribution shapes")

def plot_deaths_coverage():
    """
    Plotting fraction of true death shapes that fall inside the
    credible interval vs the credible interval width.
    """
    plot_coverage(deaths_infer_arr, deaths_pointx, "death distribution shapes")

def plot_subs_coverage():
    """
    Plotting fraction of true substitution shapes that fall inside the
    credible interval vs the credible interval width.
    """
    plot_coverage(subs_infer_arr, subs_pointx, "substitution distribution shapes")

plot_div_exp_v_true()
plot_turn_exp_v_true()
plot_births_exp_v_true()
plot_deaths_exp_v_true()
plot_subs_exp_v_true()
#plot_div_coverage()
#plot_turn_coverage()
#plot_births_coverage()
#plot_deaths_coverage()
#plot_subs_coverage()