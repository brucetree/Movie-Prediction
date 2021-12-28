import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


def get_top_feature(df_train, df_name, number):
    cast = df_train[[df_name]].values.tolist()
    cast_df = df_train[[df_name]]
    # get most popular name in cast
    count_cast_dic = {}
    for index in cast:
        item = index[0]
        real_list = eval(item)
        for single_dic in real_list:
            for k, v in single_dic.items():
                if k == 'name':
                    if v in count_cast_dic:
                        count_cast_dic[v] += 1
                    else:
                        count_cast_dic[v] = 1
    sort_count_cast = sorted(count_cast_dic.items(), key=lambda x: x[1], reverse=True)[:number]
    top_10_cast = [x[0] for x in sort_count_cast]
    # add 10 cast to column
    for i in top_10_cast:
        cast_df[i] = 0

    def cast_change(attribute, name):
        real_list = eval(attribute)
        for index in real_list:
            if index['name'] == name:
                attribute = 1
                return attribute
        attribute = 0
        return attribute

    # modify cast 10 column
    for name in top_10_cast:
        cast_df[name] = cast_df[df_name].apply(cast_change, args=(name,))
    # final version cast df
    new_cast_df = cast_df.drop(df_name, axis=1)
    return new_cast_df


def count_total_number(df_train, df_name):
    def count_spoken_languages(spoken_languages):
        real_list = eval(spoken_languages)
        number = len(real_list)
        return number

    num_cast_df = df_train[[df_name]]
    num_cast_df[df_name] = num_cast_df[df_name].apply(count_spoken_languages)
    return num_cast_df


def count_director_number(df_train, df_name, position):
    def count_number(spoken_languages):
        count_dic = {}
        real_list = eval(spoken_languages)
        for real_dic in real_list:
            for key in real_dic.keys():
                if key == 'job' and real_dic[key] == position:
                    if position in count_dic:
                        count_dic[position] += 1
                    else:
                        count_dic[position] = 1
        if position not in count_dic:
            return 0
        else:
            return count_dic[position]

    num_cast_df = df_train[[df_name]]
    num_cast_df[position] = num_cast_df[df_name].apply(count_number)
    new_num_cast_df = num_cast_df.drop(df_name, axis=1)
    return new_num_cast_df


def train_function(train_csv):
    df_train = pd.read_csv(train_csv)
    ############################################################################################
    new_cast_df = get_top_feature(df_train, 'cast', 15)
    ############################################################################################
    # get most popular name in crew
    crew = df_train[["crew"]].values.tolist()
    count_crew_dirctor_dic = {}
    count_crew_producer_dic = {}
    for index in crew:
        item = index[0]
        real_list = eval(item)
        for single_dic in real_list:
            # get director
            if single_dic['job'] == "Director":
                for k, v in single_dic.items():
                    if k == 'name':
                        if v in count_crew_dirctor_dic:
                            count_crew_dirctor_dic[v] += 1
                        else:
                            count_crew_dirctor_dic[v] = 1
            # get producer
            if single_dic['job'] == "Executive Producer":
                for k, v in single_dic.items():
                    if k == 'name':
                        if v in count_crew_producer_dic:
                            count_crew_producer_dic[v] += 1
                        else:
                            count_crew_producer_dic[v] = 1
    sort_count_crew_director = sorted(count_crew_dirctor_dic.items(), key=lambda x: x[1], reverse=True)[:10]
    sort_count_crew_producer = sorted(count_crew_producer_dic.items(), key=lambda x: x[1], reverse=True)[:10]
    top_25_director = [x[0] for x in sort_count_crew_director]
    top_25_producer = [x[0] for x in sort_count_crew_producer]

    # change director name
    new_top_25_director = []
    for index in top_25_director:
        director_name = 'director_' + index
        new_top_25_director.append(director_name)
    # add 25 director to column
    crew_director_df = df_train[["crew"]]
    for i in new_top_25_director:
        crew_director_df[i] = 0

    def director_change(director, name):
        real_list = eval(director)
        for index in real_list:
            if index['job'] == "Director" and index['name'] == name[9:]:
                director = 1
                return director
        director = 0
        return director

    # modify director 25 column
    for name in new_top_25_director:
        crew_director_df[name] = crew_director_df['crew'].apply(director_change, args=(name,))
    # final version director df
    new_director_df = crew_director_df.drop('crew', axis=1)
    # ############################################################################################
    # change producer name
    new_top_25_producer = []
    for index in top_25_producer:
        producer_name = 'producer_' + index
        new_top_25_producer.append(producer_name)
    # add 25 producer to column
    crew_producer_df = df_train[["crew"]]
    for i in new_top_25_producer:
        crew_producer_df[i] = 0

    def producer_change(producer, name):
        real_list = eval(producer)
        for index in real_list:
            if index['job'] == "Executive Producer" and index['name'] == name[9:]:
                producer = 1
                return producer
        producer = 0
        return producer

    # modify producer 25 column
    for name in new_top_25_producer:
        crew_producer_df[name] = crew_producer_df['crew'].apply(producer_change, args=(name,))
    # final version producer df
    new_producer_df = crew_producer_df.drop('crew', axis=1)
    ############################################################################################

    ############################################################################################
    # budget
    budget_df = df_train[["budget"]]
    budget_df['budget'] = budget_df['budget'].apply(lambda x: x // 1000000)

    ############################################################################################

    # ############################################################################################
    # homepage
    homepage_df = df_train[["homepage"]]
    homepage_df['homepage'] = homepage_df['homepage'].map(lambda x: 0 if pd.isna(x) else 1)

    # keyword
    keyword_df = count_total_number(df_train, 'keywords')
    # ############################################################################################
    # original_language
    def english_or_not(original_language):
        if original_language == 'en':
            original_language = 1
            return original_language
        else:
            original_language = 0
            return original_language

    original_language_df = df_train[["original_language"]]
    original_language_df['original_language'] = original_language_df['original_language'].apply(english_or_not)
    #
    # ############################################################################################
    # # production_companies
    # new_production_companies_df = get_top_feature(df_train, 'production_companies', 1)
    num_production_companies_df=count_total_number(df_train,'production_companies')
    # ############################################################################################
    # production_countries
    def US_or_not(production_countries):
        real_list = eval(production_countries)
        for index in real_list:
            if index['name'] == 'United States of America':
                production_countries = 1
                return production_countries
        production_countries = 0
        return production_countries

    production_countries_df = df_train[["production_countries"]]
    production_countries_df['production_countries'] = production_countries_df['production_countries'].apply(US_or_not)
    new_production_countries_df = get_top_feature(df_train, 'production_countries', 5)
    # ############################################################################################
    # release_date
    four_season = ['1', '2', '3', '4']
    release_date_df = df_train[["release_date"]]
    for i in four_season:
        release_date_df[i] = 0

    def release_date_change(release_date, name):
        month = int(release_date[5:7])
        if (month - 1) // 3 + 1 == int(name):
            release_date = 1
            return release_date
        release_date = 0
        return release_date

    for name in four_season:
        release_date_df[name] = release_date_df['release_date'].apply(release_date_change, args=(name,))

    new_release_date_df = release_date_df.drop('release_date', axis=1)
    # ############################################################################################
    # runtime
    # runtime_df = df_train[["runtime"]]
    # runtime_df['runtime'] = runtime_df['runtime'].apply(lambda x: int(x))
    #
    # ############################################################################################
    # spoken_languages
    spoken_languages_df = count_total_number(df_train, 'spoken_languages')
    # ############################################################################################
    # overview
    def count_str(overview):
        if type(overview) ==float:
            return 0
        list=overview.split(' ')
        return len(list)
    overview_df=df_train[['overview']]
    overview_df['overview']=overview_df['overview'].apply(count_str)
    ############################################################################################
    # tagline
    tagline_df = df_train[['tagline']]
    tagline_df['tagline'] = tagline_df['tagline'].map(lambda x: 0 if pd.isna(x) else 1)

    ############################################################################################
    num_cast_df = count_total_number(df_train, 'cast')
    # num_crew_df = count_total_number(df_train, 'crew')
    num_genres_df = count_total_number(df_train, 'genres')
    num_director_df = count_director_number(df_train, 'crew', 'Director')
    num_producer_df = count_director_number(df_train, 'crew', 'Executive Producer')
    ############################################################################################

    df_list = [new_cast_df, new_director_df, new_producer_df, budget_df, new_production_countries_df,
               new_release_date_df, num_cast_df, num_director_df, homepage_df, original_language_df,
               spoken_languages_df, num_genres_df, num_producer_df, production_countries_df,num_production_companies_df,keyword_df,overview_df,tagline_df]
    train_x_df = pd.concat(df_list, axis=1)
    train_y_df = df_train['revenue']
    # train_part2_y_df=df_train['rating']
    return train_x_df, train_y_df




def train_function_part2(train_csv):
    df_train = pd.read_csv(train_csv)
    ############################################################################################
    new_cast_df = get_top_feature(df_train, 'cast', 15)
    ############################################################################################
    # get most popular name in crew
    crew = df_train[["crew"]].values.tolist()
    count_crew_dirctor_dic = {}
    count_crew_producer_dic = {}
    for index in crew:
        item = index[0]
        real_list = eval(item)
        for single_dic in real_list:
            # get director
            if single_dic['job'] == "Director":
                for k, v in single_dic.items():
                    if k == 'name':
                        if v in count_crew_dirctor_dic:
                            count_crew_dirctor_dic[v] += 1
                        else:
                            count_crew_dirctor_dic[v] = 1
            # get producer
            if single_dic['job'] == "Executive Producer":
                for k, v in single_dic.items():
                    if k == 'name':
                        if v in count_crew_producer_dic:
                            count_crew_producer_dic[v] += 1
                        else:
                            count_crew_producer_dic[v] = 1
    sort_count_crew_director = sorted(count_crew_dirctor_dic.items(), key=lambda x: x[1], reverse=True)[:10]
    sort_count_crew_producer = sorted(count_crew_producer_dic.items(), key=lambda x: x[1], reverse=True)[:10]
    top_25_director = [x[0] for x in sort_count_crew_director]
    top_25_producer = [x[0] for x in sort_count_crew_producer]

    # change director name
    new_top_25_director = []
    for index in top_25_director:
        director_name = 'director_' + index
        new_top_25_director.append(director_name)
    # add 25 director to column
    crew_director_df = df_train[["crew"]]
    for i in new_top_25_director:
        crew_director_df[i] = 0

    def director_change(director, name):
        real_list = eval(director)
        for index in real_list:
            if index['job'] == "Director" and index['name'] == name[9:]:
                director = 1
                return director
        director = 0
        return director

    # modify director 25 column
    for name in new_top_25_director:
        crew_director_df[name] = crew_director_df['crew'].apply(director_change, args=(name,))
    # final version director df
    new_director_df = crew_director_df.drop('crew', axis=1)
    # ############################################################################################
    # change producer name
    new_top_25_producer = []
    for index in top_25_producer:
        producer_name = 'producer_' + index
        new_top_25_producer.append(producer_name)
    # add 25 producer to column
    crew_producer_df = df_train[["crew"]]
    for i in new_top_25_producer:
        crew_producer_df[i] = 0

    def producer_change(producer, name):
        real_list = eval(producer)
        for index in real_list:
            if index['job'] == "Executive Producer" and index['name'] == name[9:]:
                producer = 1
                return producer
        producer = 0
        return producer

    # modify producer 25 column
    for name in new_top_25_producer:
        crew_producer_df[name] = crew_producer_df['crew'].apply(producer_change, args=(name,))
    # final version producer df
    new_producer_df = crew_producer_df.drop('crew', axis=1)
    ############################################################################################

    ############################################################################################
    # budget
    budget_df = df_train[["budget"]]
    budget_df['budget'] = budget_df['budget'].apply(lambda x: x // 1000000)

    ############################################################################################

    # ############################################################################################
    # homepage
    homepage_df = df_train[["homepage"]]
    homepage_df['homepage'] = homepage_df['homepage'].map(lambda x: 0 if pd.isna(x) else 1)
    # keyword
    keyword_df=count_total_number(df_train,'keywords')
    #

    # ############################################################################################
    # original_language
    def english_or_not(original_language):
        if original_language == 'en':
            original_language = 1
            return original_language
        else:
            original_language = 0
            return original_language

    original_language_df = df_train[["original_language"]]
    original_language_df['original_language'] = original_language_df['original_language'].apply(english_or_not)
    #
    # ############################################################################################
    # # production_companies
    # new_production_companies_df = get_top_feature(df_train, 'production_companies', 1)
    num_production_companies_df=count_total_number(df_train,'production_companies')
    # ############################################################################################
    # production_countries
    def US_or_not(production_countries):
        real_list = eval(production_countries)
        for index in real_list:
            if index['name'] == 'United States of America':
                production_countries = 1
                return production_countries
        production_countries = 0
        return production_countries

    production_countries_df = df_train[["production_countries"]]
    production_countries_df['production_countries'] = production_countries_df['production_countries'].apply(US_or_not)
    new_production_countries_df = get_top_feature(df_train, 'production_countries', 5)
    # ############################################################################################
    # release_date
    four_season = ['1', '2', '3', '4']
    release_date_df = df_train[["release_date"]]
    for i in four_season:
        release_date_df[i] = 0

    def release_date_change(release_date, name):
        month = int(release_date[5:7])
        if (month - 1) // 3 + 1 == int(name):
            release_date = 1
            return release_date
        release_date = 0
        return release_date

    for name in four_season:
        release_date_df[name] = release_date_df['release_date'].apply(release_date_change, args=(name,))

    new_release_date_df = release_date_df.drop('release_date', axis=1)
    # ############################################################################################
    # runtime
    runtime_df = df_train[["runtime"]]
    # runtime_df['runtime'] = runtime_df['runtime'].apply(lambda x: int(x))
    #
    # ############################################################################################
    # spoken_languages
    spoken_languages_df = count_total_number(df_train, 'spoken_languages')
    # ############################################################################################

    ############################################################################################
    #
    num_cast_df = count_total_number(df_train, 'cast')
    # num_crew_df = count_total_number(df_train, 'crew')
    num_genres_df = count_total_number(df_train, 'genres')
    num_director_df = count_director_number(df_train, 'crew', 'Director')
    num_producer_df = count_director_number(df_train, 'crew', 'Executive Producer')
    ############################################################################################

    df_list = [new_cast_df, new_director_df,budget_df,new_release_date_df,num_cast_df,runtime_df,production_countries_df,num_production_companies_df,spoken_languages_df,keyword_df]
    #, new_producer_df, budget_df, new_production_countries_df,new_release_date_df, num_cast_df, num_director_df, homepage_df, original_language_df,spoken_languages_df, num_genres_df, num_producer_df, production_countries_df,num_production_companies_df
    train_x_df = pd.concat(df_list, axis=1)
    # train_y_df = df_train['revenue']
    train_part2_y_df=df_train['rating']
    return train_x_df, train_part2_y_df


if __name__ == "__main__":
    train_csv = 'training.csv'
    validation_csv = 'validation.csv'
    train_x, train_y = train_function(train_csv)
    test_x, test_y = train_function(validation_csv)

    train_x2,train_y2=train_function_part2(train_csv)
    test_x2,test_y2=train_function_part2(validation_csv)


    # PART1
    model = RandomForestRegressor(random_state=False,n_estimators=200)
    model.fit(train_x, train_y)
    Q1_predict = model.predict(test_x)
    correlation = np.corrcoef(test_y, Q1_predict)[0, 1]
    MSE = mean_squared_error(test_y, Q1_predict) / (10 ** 15)
    print(MSE)
    print("correlation: \t", correlation)

    # PART2
    # knn=KNeighborsClassifier()
    # knn.fit(train_x2,train_y2)
    # Q2_predict=knn.predict(test_x2)
    #
    # print("precision:\t", precision_score(test_y2, Q2_predict,average='macro'))
    # print("recall:\t\t", recall_score(test_y2, Q2_predict,average='macro'))
    # print("accuracy:\t", accuracy_score(test_y2, Q2_predict))