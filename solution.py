#!/usr/bin/python
###################################################################################################################
# Author: Dr. Sandeep Vidyapu (sandy.apj911@gmail.com)
# Date of Last Modification: 29.09.2022
# Summary:
# Analyze customers' ticket data and accounts' data to determine the best routing method and associated tickets
##################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, time

def data_file_names(): #Returns the accounts and tickets filenames read from the commandline
    """
    Usage:
    Python <this_python_filename.py> <accounts_filename_including_path> <tickets_filename_including_path>
    """
    if len(sys.argv) != 3:
        print(data_file_names.__doc__)
        sys.exit()

    return (sys.argv[1], sys.argv[2])


def read_data(dataF): #Returns the data read from dataF file as a pandas dataframe
    print("Trying to read data from", dataF, "...")

    try:
       data = pd.read_csv(dataF) 
    except OSError:
        print("ERROR: Unable to read data from file", dataF, "\nPlease check the path and permissions of the file and retry...")
        sys.exit()

    return data


def region_of_account_number(accDat, tickDat, accNum): #returns the region of the account number accNum
    tmpReg = accDat[accDat['account_number']==accNum]['region']
    tmpTag = tickDat[tickDat['account_number']==accNum]['tag']

    print(tmpReg)
    print(tmpTag)


#def missing_data(acc, tick): #displays the missing data
#    return None


class summaryStats: #returns summary stats on count
    def __init__(self, df, column):
        self.summary = df[column].describe()
        #print("Summary statistics of the", column, "column is:\n", self.summary)
    def add(self, metrics):
        print("Summation of", metrics, "statistics: ", sum(self.summary[metrics]))
        return sum(self.summary[metrics])


def bar_plot(df, x, y, region="Region", saveExt=".png", figTitle="region"):
    plt.figure(figsize=(8,6))
    tmpPlot = sns.barplot(data=df, x=x, y=y)
    plt.xticks(rotation=45, ha="right")
    print("Saving the plot to", figTitle, "...")
    tmpPlot.set_title(figTitle)
    figTitle = "Streaks of TAGs " + figTitle+ saveExt #filename to save
    #tmpFig = tmpPlot.get_figure()
    plt.tight_layout()
    tmpFig = plt.gcf() #get current figure
    tmpFig.savefig(figTitle)


def streak_of_tags(df, region="Region", plotStreak=True, saveExt=".png", figTitle="region"): #returns the streaks (consecutive days on which number of tags > threshold) along with its starting and ending date
    """
    Description: returns the streaks (consecutive days on which number of tags > threshold) along with its starting and ending date
    """
    df = df.copy(deep=True)
    df['tag'] = pd.to_datetime(df['tag']).dt.date #ignore the time part which is a series of zeros
    days = pd.DataFrame(columns=["start_date", "end_date", "day_count"])

    sdate = edate = None 
    count=1
    print("\nThe days on which the threshold was crossed:")
    print(df[["tag", "count"]])
    for d in df["tag"].values:
        #print(d)
        if sdate is None:
            sdate = d
        elif edate is None and pd.Timedelta(d-sdate).days!=1: #next date is not the following date
            #days=days.append({"start_date":sdate, "end_date": sdate, "day_count": 1}, ignore_index=True) #previous streak was of only one day #Deprecated
            days.loc[len(days), ["start_date", "end_date", "day_count"]] = sdate, sdate, 1 #previous streak was of only one day
            sdate = d #new start date
        elif pd.Timedelta(d-sdate).days==count: #next date is not the following date
            edate = d
            count+=1
        else:
            #days=days.append({"start_date":sdate, "end_date": edate, "day_count": count}, ignore_index=True) #Streak continutation; #Deprecated
            days.loc[len(days), ["start_date", "end_date", "day_count"]] = sdate, edate, count #streak continuation
            sdate=d
            count=1
            edate=None

    if edate is not None and pd.Timedelta(edate-sdate).days==count-1: #streak is at the end
        #days=days.append({"start_date":sdate, "end_date": edate, "day_count": count}, ignore_index=True) #deprecated
        days.loc[len(days), ["start_date", "end_date", "day_count"]] = sdate, edate, count #streak continuation

    days["day_count"] = pd.to_numeric(days["day_count"])

    if plotStreak is True:
        bar_plot(days, x="start_date", y="day_count", region=region, saveExt=saveExt, figTitle=figTitle)
        plt.close()

    print("\nStreaks (coniuous sequence of TAGs):\n", days)
    return days


def nth_largest_streak(df, n): #returns (starting_date, ending_days, day_count) of nth largest streak
    return df.nlargest(n, columns="day_count", keep="last").iloc[0]


def method_metrics(df, column="count", region="Region", metrics=None, plotStreak=True): 
    """
    Description: Determines the router method employed. Returns the used method "A" or "B" or "X"(no method)
    df: dataframe to determine the method
    column: column to use for determining the method
    plotStreak: bar plot of streaks of tags
    """
    tag_col = "tag" #name of the tag column
    streakThreshold = 7 #number of consecutive days the ticket count crosses the 'threshold' 
    ticketThreshold = 8 #onMaxTicketsMean/preMaxTicketsMean: order of raise in the tickets during max. streak

    df = df.copy(deep=True)
    df[tag_col] = pd.to_datetime(df[tag_col]).dt.date #ignore the time part which is a series of zeros
    if metrics is None:
        metrics = ["75%", "std"] #possible values 25%, mean, std, 50%, 75%, min, max

    print("\nDetermining the employed method based on", column," column on dataframe...:\n", df)
    if tag_col not in df:
        print(f"'{tag_col}' column is needed to determine the method.\nPlease provide the dataframe with the same...")
        sys.exit()

    df = df.sort_values(tag_col)
    threshold = summaryStats(df, column).add(metrics)
    print("Threshold to determine the starting of a method: ", threshold)

    tmpDF = df[df[column]>threshold]
    print("\nTAGs (dates) on which the count is greater than the threshold of", threshold, ":\n", tmpDF)
    pair_data.plot_data(pair_data, df=tmpDF, xcol="tag", ycol="count", figTitle=region+" Thresholded --> " + "Metrics: "+"+".join(metrics)+"; Threshold: "+str(int(threshold)), sortX=True, plotStats=False)

    streakDF = streak_of_tags(tmpDF, region=region, plotStreak=plotStreak, figTitle=region+" Thresholded--> " + "Metrics: "+"+".join(metrics)+"; Threshold: "+str(int(threshold)))
    sdate, edate, maxStreak = nth_largest_streak(streakDF, n=1) #largest streak
    print("\nLargest streak (> "+str(int(threshold)), " tags started on", sdate, "and ended on ", edate, ": ", str(maxStreak), "days")

    preMaxStreakMean = summaryStats(streakDF[streakDF["start_date"]<sdate], column="day_count").add(["mean"]) #based on streaks data, prior to the max streak starting date
    onMaxStreakMean = summaryStats(streakDF[streakDF["start_date"]==sdate], column="day_count").add(["mean"]) #based on streaks data, on the max streak
    fromMaxStreakMean = summaryStats(streakDF[streakDF["start_date"]>=sdate], column="day_count").add(["mean"]) #based on streaks data, from the starting of the largest streak
    preMaxTicketsMean = summaryStats(df[df["tag"]<sdate], column="count").add(["mean"]) #based on tickets data, prior to the max streak starting date
    onMaxTicketsMean = summaryStats(df[(df["tag"]>=sdate) & (df["tag"]<=edate)], column="count").add(["mean"]) #based on tickets data, prior to the max streak starting date
    fromMaxTicketsMean = summaryStats(df[df["tag"]>=sdate], column="count").add(["mean"]) #based on tickets data; from the starting date of largest streak
    print("\nPrior to max. streak starting date, average streak:", preMaxStreakMean, "days")
    print("On max. streak, average streak:", onMaxStreakMean, "days")
    print("From the max. streak starting date (including), average streak:", fromMaxStreakMean, "days")
    print("\nPrior to max. streak starting date, average ticket count:", preMaxTicketsMean)
    print("On the max. streak (from starting date to ending date), average ticket count:", onMaxTicketsMean)
    print("From the max. streak starting date(including), average ticket count:", fromMaxTicketsMean)

    returnDict = {"method": "X", "preMaxStreakMean":preMaxStreakMean, "onMaxStreakMean":onMaxStreakMean, "fromMaxStreakMean":fromMaxStreakMean, "preMaxTicketsMean":preMaxTicketsMean, "onMaxTicketsMean":onMaxTicketsMean, "fromMaxTicketsMean":fromMaxTicketsMean}
    if onMaxStreakMean < streakThreshold: #streak was not even for 7 days; no router method was employed
        print(f"\nStreak({onMaxStreakMean}) was note even for {streakThreshold} days. So, returning as method X (no method)...")
        returnDict["method"] = "X"
    else:
        if onMaxTicketsMean/preMaxTicketsMean > ticketThreshold:
            print(f"\nonMaxTicketsMean/preMaxTicketsMean ({onMaxTicketsMean/preMaxTicketsMean}) is greater than the ticketThreshold ({ticketThreshold}). So, returning as method B...")
            returnDict["method"] = "B"
        else:
            print(f"\nonMaxTicketsMean/preMaxTicketsMean ({onMaxTicketsMean/preMaxTicketsMean}) is less than or equal to the ticketThreshold ({ticketThreshold}). So, returning as method A (staggered rollout)...")
            returnDict["method"] = "A" #Staggered rollout
        
    return returnDict

class pair_data: #pairs the account data and tickets data based on account number
    def __init__(self, acc, tick):
        self.acc = acc #accounts data with corresponding entry in ticket data
        self.tick = tick #tickets data with corresponding entry in account data

        self.accNtick = self.acc.merge(self.tick, on="account_number", how="inner") #paired account and tickets data
        self.accNOtag = self.acc.merge(self.tick, on="account_number", how="left") #account numbers with NO tag (date info)
        self.accNOtag = self.accNOtag[self.accNOtag.isna().any(axis=1)]

        self.accNtick = self.accNtick.sort_values(["account_number", "tag"], ascending=[True, True]) #sort the records by tag followed by account_number
        self.accNOtag = self.accNOtag.sort_values(["account_number", "tag"], ascending=[True, True]) #sort the records by tag followed by account_number

    def disp_pair_data_stats(self): #prints the basic statistics of the data
        print("\nPaired (accounts and tickets) data records:\n", self.accNtick)
        print("\nRecords/accounts with NO TAG DATA:\n", self.accNOtag)
        print("\nNumber of paired (accounts and tickets) data records:", self.accNtick.shape[0])
        print("Number of UNIQUE account numbers in paired data:", len(self.accNtick["account_number"].unique()))
        print("Account number(s) with the MOST TICKETS in paired data:", list(self.accNtick.mode()["account_number"]))
        print("TAG (date) with highest frequency (most tickets) in paired data:", list(self.accNtick.mode()["tag"]))
        print("Account number(s) with the MOST TICKETS in paired data:\n", self.accNtick.value_counts())
        print("\nAccount numbers with generated TICKETS count (in decreasing order) in paired data:\n", self.accNtick.value_counts(subset="account_number"))
        print("\nTAGs (dates) frequency (in decreasing order) in paired data:\n", self.accNtick.value_counts(subset="tag"))

        print("\n\nNumber of records/accounts with NO TAG DATA:", self.accNOtag.shape[0])
        print("Number of UNIQUE account numbers with NO TAG DATA:", len(self.accNOtag["account_number"].unique()))
        print("Account number(s) with the MOST TICKETS WITHOUT TAGs (dates):", list(self.accNOtag.mode()["account_number"]))
        print("\nAccount numbers with generated TICKETS count (in decreasing order) in records with NO TAG DATA\n", self.accNOtag.value_counts(subset="account_number"))

    def frequency_dataframe(self, df, column): #returns the frequency of the values in given column
        tmp = df.value_counts(subset=column) #frequency as pandas Series
        tmp = pd.DataFrame({column: tmp.index, 'count': tmp.values}) #series to dataframe
        print("\nFrequency (count) of values in", column, "column is:\n", tmp)
        return tmp

    def smart_plot_data(func):
        def plotting_data(cls, df, xcol, ycol, figTitle=None, sortX=False, saveExt=".png", plotStats=True): #plots the data from xcol Vs. ycol; sortX: sort the xcol before plotting
            df = df.copy(deep=True)
            if figTitle is None: # save the figure to a filename starting with "tmp"
                figTitle = "tmp"
            if sortX is True: #sort the records based on values in xcol
                df = df.sort_values(xcol)
            if "tag" in df: #plotting becomes easier by converting the tag column into datetime format
                df['tag'] = pd.to_datetime(df['tag'])
            plt.figure(figsize=(8,6))
            tmpPlot = sns.lineplot(data=df, x=xcol, y=ycol, label="daily")
            #tmpPlot = sns.barplot(data=df, x=xcol, y=ycol)
            plt.xticks(rotation=45, ha="right")
            tmpPlot.set_title(figTitle)
            return func(cls, df, xcol, ycol, figTitle, sortX, saveExt, plotStats)
        return plotting_data

    @smart_plot_data
    def plot_data(cls, df, xcol, ycol, figTitle, sortX, saveExt, plotStats): #plots the data from xcol Vs. ycol; and also, plots running average of counts

        if plotStats is True: #Plot other statistics including mean, 25%, 50%
            window = 7 #number of days to consider for moving window

            df["moving_average"] = df["count"].rolling(window=window).mean()
            print("\nMoving averaged count/freuqncy for "+str(window)+" days:\n", df)
            tmpPlot = sns.lineplot(data=df, x=xcol, y="moving_average", label="moving average of "+str(window)+" days")
            
            #plot summary statistics
            stats = summaryStats(df, column="count")
            tmpPlot.axhline(stats.add(["mean"]), color="black", linestyle=":", label="mean")
            tmpPlot.axhline(stats.add(["50%"]), color="b", linestyle="-", label="50%")
            tmpPlot.axhline(stats.add(["75%"]), color="g", linestyle="-.", label="75%")
            tmpPlot.axhline(stats.add(["mean", "std"]), color="r", linestyle="--", label="mean+std")
            tmpPlot.axhline(stats.add(["50%", "std"]), color="y", linestyle="-.", label="50%+std")
            tmpPlot.axhline(stats.add(["75%", "std"]), color="m", linestyle="-", label="75%+std")
        
        #save the plot to a file
        plt.legend(loc ='best')
        plt.tight_layout()
        figTitle = figTitle + saveExt
        print("Saving the plot to", figTitle, "...")
        #tmpFig = tmpPlot.get_figure()
        tmpFig = plt.gcf() #get current figure
        tmpFig.savefig(figTitle)
        plt.close()


def additional_tickets(rs, rt): #returns the number of additional tickets expected due to the rollong of method A or method B.
    """
    Description: provides the expected additional tickets in total and on per day basis
    rs: source region: region on which we need to compute the expected additional tickets
    rt: target region: region on which the method was already applied/piloted to observe the effect
    """
    addTicks_streakEffectOnly = 0 #Additional tickets by utilizing only the effect observed during max streak
    addTicks_streakNpotEffect = 0 #Additional tickets by utilizing the effect observed during and post the max streak

    addTicks_streakEffectOnly = rs["preMaxTicketsMean"]*(rt["onMaxTicketsMean"]/rt["preMaxTicketsMean"])*rt["onMaxStreakMean"] - rs["preMaxTicketsMean"]*rt["onMaxStreakMean"] #scale the expected tickets based on observed spike
    addTicks_streakNpotEffect = rs["preMaxTicketsMean"]*(rt["fromMaxTicketsMean"]/rt["preMaxTicketsMean"])*rt["fromMaxStreakMean"] - rs["preMaxTicketsMean"]*rt["fromMaxStreakMean"] #scale the expected tickets based on observed and post spikes
    addTickets = {"streakOnlyTotal": int(addTicks_streakEffectOnly), "streakOnlyPerDay": int(addTicks_streakEffectOnly/rt["onMaxStreakMean"]), "streakOnlyDayCount": int(rt["onMaxStreakMean"]),"streakNpostTotal": int(addTicks_streakNpotEffect), "streakNpostPerDay": int(addTicks_streakNpotEffect/rt["fromMaxStreakMean"]), "streakNpostDayCount": int(rt["fromMaxStreakMean"])}
    print(f"Additional Tickets: {addTickets}")

    return addTickets


def analysis(acc, tick):
    """
    Description: Analyzes, determines the method applied in each region and then provides the expected additional costs with the application of method-A and method-B
    acc: accounts data as pandas dataframe
    tick: tickets data as pandas dataframe
    """
    print("\nAccounts data (as given):\n", acc)
    print("\nTickets data (as given):\n", tick)
    acc = acc.sort_values("account_number")
    acc = acc.sort_values("region")
    tick = tick.sort_values("tag")
    tick = tick.sort_values("account_number")

    regionX = [] #regions where method X (no method) was utilized
    regionA = [] #regions where method A (staggered method) was utilized
    regionB = [] #regions where method B was utilized
    for r in acc['region'].unique():
        print("\n============================================ DETERMINING THE APPLIED METHOD in", r, "===========================================================")
        acc_reg = acc[acc['region']==r]
        #print("\nAccounts data in", r, ":\n", acc_reg)
        tmp = pair_data(acc_reg, tick)
        print("\nPaired account and ticket data:\n", tmp.accNtick)
        print("\nMissing data in pairing the account and ticket data:\n", tmp.accNOtag)

        print(tmp.disp_pair_data_stats())
        tmpFreq = tmp.frequency_dataframe(df=tmp.accNtick, column="tag")
        tmp.plot_data(df=tmpFreq, xcol="tag", ycol="count", figTitle=r, sortX=True)
        #regionDict = method_metrics(tmpFreq, region=r, metrics=["50%", "std"])
        regionDict = method_metrics(tmpFreq, region=r, metrics=["75%", "std"])
        regionDict["totalDays"] = tmpFreq.shape[0] #total number of days for which the data was recorded
        regionDict["totalTickets"] = sum(tmpFreq["count"]) #total number of tickets recorded for the considered days
        regionDict["region"] = r
        print(f"{r} was employed with method {regionDict['method']}")

        if regionDict["method"] == "X":
            regionX.append(regionDict)
        elif regionDict["method"] == "A":
            regionA.append(regionDict)
        else:
            regionB.append(regionDict)

    print("\n============================================ COMPUTING THE ADDITIONAL TICKETS BY APPLYING METHOD-A and METHOD-B ===========================================================")
    for rx in regionX:
        for ra in regionA:
            print(f"\n-------------------- APPLYING METHOD-A in {rx['region']} that was applied in {ra['region']}--------------------------")
            additional_tickets(rx, ra)
        for rb in regionB:
            print(f"\n-------------------- APPLYING METHOD-B in {rx['region']} that was applied in {rb['region']}--------------------------")
            additional_tickets(rx, rb)


if __name__ == '__main__':
    accountsF, ticketsF = data_file_names()
    accounts = read_data(accountsF)
    tickets = read_data(ticketsF)
    analysis(accounts, tickets)

