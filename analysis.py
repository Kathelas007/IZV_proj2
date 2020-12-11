#!/usr/bin/env python3.8
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle

int_columns = ['p1', 'p36', 'p37', 'weekday(p2a)', 'p6', 'r', 's']
float_columns = ['o', ]
str_columns = ['h', 'i', 'k', 'l', 'n', 'p', 'q', 't']


# Ukol 1: nacteni dat
def get_dataframe(filename: str = "accidents.pkl.gz", verbose: bool = False) -> pd.DataFrame:
    get_memory = lambda x: x.memory_usage(deep=True).sum() / (1024 ** 2)

    adf = pd.read_pickle(filename)

    if verbose:
        print('orig_size={:.1f} MB'.format(get_memory(adf)))

    # date
    adf['date'] = adf['p2a']
    adf['date'] = pd.to_datetime(adf['date'])

    adf = adf.drop(['p2a', 'p2b', 'j'], axis=1)

    for col in str_columns:
        adf[col] = adf[col].astype(str)

    for col in int_columns:
        adf[col] = pd.to_numeric(adf[col])

    for col in float_columns:
        adf[col] = pd.to_numeric(adf[col].str.replace(',', '.'))

    if verbose:
        print('new_size={:.1f} MB'.format(get_memory(adf)))

    return adf


# Ukol 2: následky nehod v jednotlivých regionech
def plot_conseq(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    df_plot = df[['p1', 'p13a', 'p13b', 'p13c', 'region']]

    df_plot = df_plot.groupby(['region']).agg(
        death=('p13a', 'sum'),
        hard=('p13b', 'sum'),
        light=('p13c', 'sum'),
        total=('p1', 'count')
    ).sort_values(by=['total', ], axis=0, ascending=False).reset_index()

    axes_names = ["Úmrtí", "Těžká zranění", "Lehká zranění", "Celkem nehod"]

    sns.set_style("darkgrid")
    palette = sns.dark_palette("#69d", reverse=False, n_colors=14)

    fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69), sharex=True)
    fig.suptitle("Následky nehod v jednotlivých krajích", fontsize=16)
    axes = axes.flatten()

    for i in range(4):
        column = df_plot.columns[i + 1]
        rank = df_plot[column].argsort()

        sns.barplot(data=df_plot, x="region", y=column, ax=axes[i], palette=np.array(palette[::-1])[rank])

        axes[i].set(ylabel="Počet", xlabel='')
        axes[i].set_title(axes_names[i])
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

    axes[-1].set(xlabel="Kraj")

    if show_figure:
        plt.show()
        plt.close()

    if fig_location is not None:
        fig.savefig(fig_location)


# Ukol 3: příčina nehody a škoda
def plot_damage(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    # select only desired data
    selected_regions = ["PHA", "STC", "ULK", "JHM"]
    df_plot = df[['p12', 'p53', 'region']]
    df_plot = df_plot[df_plot["region"].isin(selected_regions)]
    df['p53'] = df['p53'].div(10).astype(int)

    # couple pricina nehody
    p12_interval = [(100, 100), (201, 209), (301, 311), (401, 414), (501, 516), (601, 615)]
    p12_labels = ["nezaviněná řidičem", "nepřiměřená rychlost jízdy", "nesprávné předjíždění",
                  "nedání přednosti v jízdě", "nesprávný způsob jízdy", "technická závada vozidla"]

    intervals = pd.IntervalIndex.from_tuples(p12_interval, closed="both")
    df_plot['p12'] = pd.cut(df_plot['p12'], bins=intervals).map(
        dict(zip(intervals, p12_labels)))

    # couple skoda
    max_value = df_plot['p53'].max()
    p53_intervals = [-1, 50, 200, 500, 1000, max_value + 1]
    p53_labels = ["<50", "50-200", "200-500", "500-1000", ">1000"]
    df_plot['p53'] = pd.cut(df_plot['p53'], p53_intervals, labels=p53_labels)

    # plot
    sns.set_style("darkgrid")
    sns.set_palette(sns.color_palette("hls"))

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.subplots_adjust(bottom=0.8)
    fig.suptitle("Příčina nehody a škoda v krajích\nPraha, Středočeský, Ústecký a Olomoucký", fontsize=14,
                 fontweight='bold')
    axes = axes.flatten()

    fig.tight_layout(rect=(0.5, 0.5, 1, 1))

    max_y_value = df_plot.groupby(['region', 'p12'])['p53'].value_counts().max()
    for i in range(4):
        data = df_plot[df_plot["region"] == selected_regions[i]]
        sns.countplot(data=data, x="p53", hue="p12", ax=axes[i])

        axes[i].set(yscale="log")
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].set(ylabel="Škoda [tisíc Kč]", xlabel='Počet')
        axes[i].get_legend().remove()
        axes[i].set_title(selected_regions[i])
        axes[i].set_ylim(top=max_y_value * 1.2)
        axes[i].tick_params(axis='x', labelrotation=10)

    # legend
    # handles = labels = []
    # for i in range(4):
    #     handle, label = axes[i].get_legend_handles_labels()
    #     handles.append(handle)
    #     labels.append(label)

    #handles, labels = axes[3].get_legend_handles_labels()
    fig.tight_layout()
    #fig.legend(ncol=2, loc=(0, -0.2))
    # fig.legend(handles, p12_labels, bbox_to_anchor=(0, 1), loc='upper left', ncol=2)
    axes[2].legend(loc=(0, -0.55))

    if show_figure:
        plt.show()
        plt.close()

    if fig_location is not None:
        fig.savefig(fig_location)


# Ukol 4: povrch vozovky
def plot_surface(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    pass


if __name__ == "__main__":
    pass
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni ¨
    # funkce.
    df = get_dataframe("accidents.pkl.gz")
    # plot_conseq(df, fig_location="01_nasledky.png", show_figure=True)
    plot_damage(df, "02_priciny.png", True)
    # plot_surface(df, "03_stav.png", True)
