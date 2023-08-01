import pandas as pd

# df = pd.read_csv("r02 - 02.2e.csv")
# df2 = df.mean(axis=0)

# df3 = df2.to_numpy()

def decision_tree_eating(filename):
    df = pd.read_csv(filename + ".csv")

    df = df.mean(axis = 0)
    print(df)
    df = df.to_numpy()
    print(df)

    if df[2] <= 19.47584:
        if df[13] <= 176.793777:
            if df[24] <= 205.1583:
                if df[19] > 70.03324:
                    if df[1] <= 7.667537:
                        if df[9] > 37.724317:
                            if df[22] <= 36.00698:
                                if df[7] > 30.778984:
                                    if df[1] > 7.147857:
                                        x = "E3"
                                    else:
                                        x = "E2"
                                else:
                                    x = "E1"
                            else:
                                x = "E1"
                        else:
                            x = "E2"
                    else:
                        x = "E2"
                else:
                    x = "E3"
            else:
                x = "E1"
        else:
            x = "E2"
    else:
        if df[3] <= 39.148446:
            if df[5] > 26.814816:
                if df[18] > 62.620915:
                    if df[13] <= 250.278336:
                        x = "E1"
                    else:
                        x = "E3"
                else:
                    if df[13] <= 179.331012:
                        if df[15] <= 40.09142:
                            if df[21] <= 115.9192:
                                if df[12] > 50.244878:
                                    if df[11] <= 41.730896:
                                        x = "E1"
                                    else:
                                        x = "E3"
                                else:
                                    x = "E3"
                            else:
                                x = "E2"
                        else:
                            if df[18] <= 27.929134:
                                if df[8] > 30.339747:
                                    if df[2] <= 24.244605:
                                        x = "E1"
                                    else:
                                        x = "E2"
                                else:
                                    x = "E3"
                            else:
                                if df[13] > 141.69176:
                                    if df[24] <= 216.1081:
                                        if df[7] > 35.123149:
                                            if df[4] <= 17.741172:
                                                if df[2] <= 22.783822:
                                                    x = "E3"
                                                else:
                                                    x = "E2"
                                            else:
                                                x = "E3"
                                        else:
                                            x = "E2"
                                    else:
                                        if df[1] <= 10.263676:
                                            if df[24] <= 217.6182:
                                                x = "E1"
                                            else:
                                                x = "E3"
                                        else:
                                            if df[1] <= 11.263923:
                                                x = "E2"
                                            else:
                                                x = "E3"
                                else:
                                    x = "E3"
                    else:
                        if df[5] <= 43.027079:
                            if df[19] <= 126.3204:
                                x = "E2"
                            else:
                                x = "E1"
                        else:
                            if df[24] <= 379.555787:
                                x = "E3"
                            else:
                                x = "E2"

            else:
                x = "E2"
        else:
            x = "E2"

    return x



if __name__ == "__main__":
    classification = decision_tree_eating("r02 - 02.2e")
    print(f"Classified eating action: {classification}")
