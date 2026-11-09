# NOTAS
## Columnas incompletas
| Cantidad | Porcentaje | Columna                                | Observaciones          |
| -------: | ---------: | -------------------------------------- | ---------------------- |
|     199k |       100% | Voluntary Soft-Story Retrofit          |                        |
|     199k |       100% | TIDF Compliance                        |                        |
|     197k |        99% | Street Number Suffix                   |                        |
|     197k |        99% | Unit Suffix                            |                        |
|     194k |        97% | Site Permit                            | Mismatched: 5359 (3%)  |
|     192k |        97% | Structural Notification                | Mismatched: 6922 (3%)  |
|     180k |        91% | Fire Only Permit                       | Mismatched: 18.8k (9%) |
|     169k |        85% | Unit                                   |                        |
|     102k |        51% | Completed Date                         |                        |
|    51.9k |        26% | Permit Expiration Date                 |                        |
|    51.5k |        26% | Existing Units                         |                        |
|    50.9k |        26% | Proposed Units                         |                        |
|    42.8k |        22% | Number of Existing Stories             |                        |
|    42.9k |        22% | Number of Proposed Stories             |                        |
|    43.4k |        22% | Existing Construction Type             |                        |
|    43.4k |        22% | Existing Construction Type Description |                        |
|    43.2k |        22% | Proposed Construction Type             |                        |
|    43.2k |        22% | Proposed Construction Type Description |                        |
|    41.1k |        21% | Existing Use                           |                        |
|    42.4k |        21% | Proposed Use                           |                        |
|    38.1k |        19% | Estimated Cost                         |                        |
|    37.3k |        19% | Plansets                               |                        |
|    14.9k |         8% | Issued Date                            |                        |
|    14.9k |         8% | First Construction Document Date       |                        |
|     6066 |         3% | Revised Cost                           |                        |
|     2768 |         1% | Street Suffix                          |                        |
|     1717 |         1% | Supervisor District                    |                        |
|     1725 |         1% | Neighborhoods - Analysis Boundaries    |                        |
|     1716 |         1% | Zipcode                                |                        |
|     1700 |         1% | Location                               |                        |
|      290 |         0% | Description                            |                        |

## Datos altamente correlacionados

| Variable 1                 | Variable 2                 | Correlación | Ambas con dato | Solo V1 nula | Solo V2 nula | Ambas nulas |
| -------------------------- | -------------------------- | ----------: | -------------: | -----------: | -----------: | ----------: |
| Number of Existing Stories | Number of Proposed Stories |      0.9974 |         152622 |         3410 |         3494 |       39374 |
| Estimated Cost             | Revised Cost               |      0.9788 |         155373 |        37461 |         5461 |         605 |
| Existing Units             | Proposed Units             |      0.9941 |         144974 |         3015 |         2388 |       48523 |
| Existing Construction Type | Proposed Construction Type |      0.9990 |         151995 |         3743 |         3539 |       39623 |
> Para rellenar los datos faltantes aplicare una regresión lineal
### Fechas


| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana | Solo Fecha 1 nula | Solo Fecha 2 nula | Ambas nulas |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: | ----------------: | ----------------: | ----------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |                 0 |            101709 |           0 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |                26 |                32 |       14914 |
| Issued Date                      | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |                20 |             36960 |       14920 |
| First Construction Document Date | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |                30 |             36964 |       14916 |
| Filed Date                       | Permit Expiration Date           |                      17.0000 |       360.0000 |        0.0472 |                 0 |             51880 |           0 |
| Current Status Date              | Permit Expiration Date           |                     157.0000 |       322.0000 |        0.4876 |                 0 |             51880 |           0 |
| Completed Date                   | Permit Expiration Date           |                     161.0000 |       281.0000 |        0.5730 |             49862 |                33 |       51847 |
| Issued Date                      | Completed Date                   |                     181.0000 |       100.0000 |        1.8100 |                19 |             86788 |       14921 |
| Completed Date                   | First Construction Document Date |                     180.0000 |        99.0000 |        1.8182 |             86805 |                42 |       14904 |
| Filed Date                       | Completed Date                   |                     208.0000 |       112.0000 |        1.8571 |                 0 |            101709 |           0 |
| Current Status Date              | Filed Date                       |                     146.0000 |        24.0000 |        6.0833 |                 0 |                 0 |           0 |
| Current Status Date              | First Construction Document Date |                     119.0000 |        13.0000 |        9.1538 |                 0 |             14946 |           0 |
| Current Status Date              | Issued Date                      |                     116.0000 |        11.0000 |       10.5455 |                 0 |             14940 |           0 |
| Filed Date                       | Issued Date                      |                       6.0000 |         0.0000 |           inf |                 0 |             14940 |           0 |
| Filed Date                       | First Construction Document Date |                       6.0000 |         0.0000 |           inf |                 0 |             14946 |           0 |


![img.png](img.png)

![img_1.png](img_1.png)


## Fechas por Permit Type

### Permit Type: 1
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Issued Date                      |                       0.0000 |         0.0000 |        0.0000 |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                     128.7500 |      2160.0000 |        0.0596 |
| Filed Date                       | Permit Expiration Date           |                     344.5000 |      2501.0000 |        0.1377 |
| First Construction Document Date | Permit Expiration Date           |                     453.2500 |      2007.5000 |        0.2258 |
| Current Status Date              | Permit Expiration Date           |                     791.0000 |      2160.0000 |        0.3662 |
| Completed Date                   | First Construction Document Date |                     268.0000 |       690.0000 |        0.3884 |
| Issued Date                      | Completed Date                   |                     306.0000 |       715.0000 |        0.4280 |
| Filed Date                       | Completed Date                   |                     533.0000 |      1121.0000 |        0.4755 |
| Filed Date                       | Issued Date                      |                     305.5000 |       419.5000 |        0.7282 |
| Completed Date                   | Permit Expiration Date           |                     860.0000 |      1168.0000 |        0.7363 |
| Filed Date                       | First Construction Document Date |                     391.2500 |       525.0000 |        0.7452 |
| Current Status Date              | Filed Date                       |                     615.0000 |       301.0000 |        2.0432 |
| Issued Date                      | First Construction Document Date |                     157.7500 |        70.5000 |        2.2376 |
| Current Status Date              | First Construction Document Date |                     329.0000 |       140.0000 |        2.3500 |

### Permit Type: 2
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                       0.0000 |      1080.0000 |        0.0000 |
| First Construction Document Date | Permit Expiration Date           |                     215.0000 |      1045.0000 |        0.2057 |
| Completed Date                   | Permit Expiration Date           |                     264.0000 |       724.0000 |        0.3646 |
| Filed Date                       | Permit Expiration Date           |                     616.0000 |      1383.0000 |        0.4454 |
| Current Status Date              | Permit Expiration Date           |                     398.0000 |       828.0000 |        0.4807 |
| Completed Date                   | First Construction Document Date |                     226.0000 |       374.0000 |        0.6043 |
| Issued Date                      | Completed Date                   |                     313.0000 |       393.0000 |        0.7964 |
| Filed Date                       | Completed Date                   |                     577.0000 |       526.0000 |        1.0970 |
| Current Status Date              | First Construction Document Date |                     282.0000 |       252.0000 |        1.1190 |
| Filed Date                       | First Construction Document Date |                     567.0000 |       475.0000 |        1.1937 |
| Filed Date                       | Issued Date                      |                     451.0000 |       303.0000 |        1.4884 |
| Current Status Date              | Filed Date                       |                     688.5000 |       379.5000 |        1.8142 |
| Current Status Date              | Issued Date                      |                     402.0000 |       221.0000 |        1.8190 |
| Issued Date                      | First Construction Document Date |                     182.0000 |        35.0000 |        5.2000 |

### Permit Type: 3
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |
| Filed Date                       | Permit Expiration Date           |                     793.5000 |       870.0000 |        0.9121 |
| Filed Date                       | Completed Date                   |                     429.5000 |       470.0000 |        0.9138 |
| Filed Date                       | Issued Date                      |                     229.0000 |       203.0000 |        1.1281 |
| Completed Date                   | First Construction Document Date |                     311.2500 |       266.0000 |        1.1701 |
| Issued Date                      | Completed Date                   |                     343.0000 |       280.0000 |        1.2250 |
| Filed Date                       | First Construction Document Date |                     273.5000 |       223.0000 |        1.2265 |
| Completed Date                   | Permit Expiration Date           |                     466.2500 |       295.5000 |        1.5778 |
| Current Status Date              | Permit Expiration Date           |                     674.5000 |       365.0000 |        1.8479 |
| Issued Date                      | Permit Expiration Date           |                     720.0000 |       365.0000 |        1.9726 |
| First Construction Document Date | Permit Expiration Date           |                     720.0000 |       365.0000 |        1.9726 |
| Current Status Date              | Filed Date                       |                     488.0000 |       234.0000 |        2.0855 |
| Current Status Date              | First Construction Document Date |                     310.0000 |       120.0000 |        2.5833 |
| Current Status Date              | Issued Date                      |                     314.0000 |        57.0000 |        5.5088 |

### Permit Type: 4
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |
| First Construction Document Date | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |
| Filed Date                       | Permit Expiration Date           |                      37.0000 |       367.0000 |        0.1008 |
| Current Status Date              | Permit Expiration Date           |                     158.7500 |       347.0000 |        0.4575 |
| Completed Date                   | Permit Expiration Date           |                     163.0000 |       242.0000 |        0.6736 |
| Filed Date                       | Completed Date                   |                     184.0000 |       157.0000 |        1.1720 |
| Issued Date                      | Completed Date                   |                     159.0000 |       118.0000 |        1.3475 |
| Completed Date                   | First Construction Document Date |                     159.0000 |       118.0000 |        1.3475 |
| Current Status Date              | Filed Date                       |                     174.0000 |        41.0000 |        4.2439 |
| Filed Date                       | Issued Date                      |                      48.0000 |        10.0000 |        4.8000 |
| Filed Date                       | First Construction Document Date |                      48.0000 |        10.0000 |        4.8000 |
| Current Status Date              | Issued Date                      |                      99.0000 |         0.0000 |           inf |
| Current Status Date              | First Construction Document Date |                      99.0000 |         0.0000 |           inf |

### Permit Type: 5
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                     720.0000 |      1080.0000 |        0.6667 |
| First Construction Document Date | Permit Expiration Date           |                     720.0000 |      1080.0000 |        0.6667 |
| Filed Date                       | Permit Expiration Date           |                     795.0000 |      1100.0000 |        0.7227 |
| Filed Date                       | Completed Date                   |                     710.0000 |       835.0000 |        0.8503 |
| Current Status Date              | Permit Expiration Date           |                     572.5000 |       610.0000 |        0.9385 |
| Issued Date                      | Completed Date                   |                     746.2500 |       737.5000 |        1.0119 |
| Completed Date                   | First Construction Document Date |                     746.2500 |       737.5000 |        1.0119 |
| Filed Date                       | Issued Date                      |                      81.0000 |        68.5000 |        1.1825 |
| Filed Date                       | First Construction Document Date |                      81.0000 |        67.0000 |        1.2090 |
| Completed Date                   | Permit Expiration Date           |                     565.5000 |       370.0000 |        1.5284 |
| Current Status Date              | Filed Date                       |                     757.0000 |       246.0000 |        3.0772 |
| Current Status Date              | Issued Date                      |                     777.5000 |       139.5000 |        5.5735 |
| Current Status Date              | First Construction Document Date |                     775.0000 |       125.0000 |        6.2000 |

### Permit Type: 6
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                       0.0000 |       180.0000 |        0.0000 |
| First Construction Document Date | Permit Expiration Date           |                       0.0000 |       180.0000 |        0.0000 |
| Current Status Date              | Permit Expiration Date           |                      61.0000 |       180.0000 |        0.3389 |
| Issued Date                      | Completed Date                   |                     233.5000 |       217.0000 |        1.0760 |
| Completed Date                   | First Construction Document Date |                     233.5000 |       217.0000 |        1.0760 |
| Filed Date                       | Permit Expiration Date           |                     472.0000 |       422.0000 |        1.1185 |
| Filed Date                       | Completed Date                   |                     577.7500 |       483.0000 |        1.1962 |
| Completed Date                   | Permit Expiration Date           |                     211.0000 |       101.5000 |        2.0788 |
| Filed Date                       | Issued Date                      |                     466.0000 |       218.0000 |        2.1376 |
| Filed Date                       | First Construction Document Date |                     466.0000 |       218.0000 |        2.1376 |
| Current Status Date              | Filed Date                       |                     595.2500 |       256.0000 |        2.3252 |
| Current Status Date              | First Construction Document Date |                     217.0000 |        11.0000 |       19.7273 |
| Current Status Date              | Issued Date                      |                     217.0000 |         0.0000 |           inf |

### Permit Type: 7
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |
| First Construction Document Date | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |
| Filed Date                       | Permit Expiration Date           |                      36.5000 |       364.0000 |        0.1003 |
| Current Status Date              | Permit Expiration Date           |                     116.7500 |       360.0000 |        0.3243 |
| Completed Date                   | Permit Expiration Date           |                     139.5000 |       293.0000 |        0.4761 |
| Filed Date                       | Completed Date                   |                     156.0000 |        88.5000 |        1.7627 |
| Issued Date                      | Completed Date                   |                     103.2500 |        55.5000 |        1.8604 |
| Completed Date                   | First Construction Document Date |                     108.0000 |        52.0000 |        2.0769 |
| Filed Date                       | Issued Date                      |                      48.0000 |         9.0000 |        5.3333 |
| Filed Date                       | First Construction Document Date |                      48.0000 |         9.0000 |        5.3333 |
| Current Status Date              | Filed Date                       |                      98.0000 |        14.0000 |        7.0000 |
| Current Status Date              | Issued Date                      |                      14.5000 |         0.0000 |           inf |
| Current Status Date              | First Construction Document Date |                      13.0000 |         0.0000 |           inf |

### Permit Type: 8
| Fecha 1                          | Fecha 2                          | Rango intercuartilico (dias) | Mediana (dias) | IQR / Mediana |
| -------------------------------- | -------------------------------- | ---------------------------: | -------------: | ------------: |
| Current Status Date              | Completed Date                   |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | First Construction Document Date |                       0.0000 |         0.0000 |        0.0000 |
| Issued Date                      | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |
| First Construction Document Date | Permit Expiration Date           |                       0.0000 |       360.0000 |        0.0000 |
| Filed Date                       | Permit Expiration Date           |                       8.0000 |       360.0000 |        0.0222 |
| Current Status Date              | Permit Expiration Date           |                     160.0000 |       317.0000 |        0.5047 |
| Completed Date                   | Permit Expiration Date           |                     158.0000 |       281.0000 |        0.5623 |
| Completed Date                   | First Construction Document Date |                     166.0000 |        93.0000 |        1.7849 |
| Filed Date                       | Completed Date                   |                     183.0000 |       102.0000 |        1.7941 |
| Issued Date                      | Completed Date                   |                     167.0000 |        93.0000 |        1.7957 |
| Current Status Date              | Filed Date                       |                     120.0000 |        19.0000 |        6.3158 |
| Current Status Date              | Issued Date                      |                     110.0000 |        11.0000 |       10.0000 |
| Current Status Date              | First Construction Document Date |                     110.0000 |        11.0000 |       10.0000 |
| Filed Date                       | Issued Date                      |                       3.0000 |         0.0000 |           inf |
| Filed Date                       | First Construction Document Date |                       3.0000 |         0.0000 |           inf |

![plot_2026-09-09 15-20-23_0.png](Adjuntos/plot_2026-09-09%2015-20-23_0.png)
![plot_2026-09-09 15-20-23_1.png](Adjuntos/plot_2026-09-09%2015-20-23_1.png)
![plot_2026-09-09 15-20-23_2.png](Adjuntos/plot_2026-09-09%2015-20-23_2.png)
![plot_2026-09-09 15-20-23_3.png](Adjuntos/plot_2026-09-09%2015-20-23_3.png)
![plot_2026-09-09 15-20-23_4.png](Adjuntos/plot_2026-09-09%2015-20-23_4.png)
![plot_2026-09-09 15-20-23_5.png](Adjuntos/plot_2026-09-09%2015-20-23_5.png)
![plot_2026-09-09 15-20-23_6.png](Adjuntos/plot_2026-09-09%2015-20-23_6.png)
![plot_2026-09-09 15-20-23_7.png](Adjuntos/plot_2026-09-09%2015-20-23_7.png)

> bajo esto inferimos que el permit type es muy significativo para intentar predecir las fechas de los eventos teniendo varianzas en los IQR / Mediana centrandonos en la mediana dado que que es el valor que propongo para sustituir aun que ciertamente los outlayers siguen siendo numerosos por lo que temo que existan variables que no estoy tomando en cuenta o quizas estoy sobreestimando los datos dado que no había trabajado con un dataset tan grande

> Inputare los datos calculando la fecha mediante el tipo de permiso introducire la fecha a la distancia media entre la fecha de referencia

```python
import pandas as pd
df = pd.read_csv('Hoja de cálculo sin título - Sheet1.csv')
print(df)


```

```text
    Sl No                             Column name                                                                            Description
0       1                           Permit Number                                                           Number assigned while filing
1       2                             Permit Type                                            Type of the permit represented numerically.
2       3                  Permit Type Definition            Description of the Permit type, for example\n new construction, alterations
3       4                    Permit Creation Date                      Date on which permit created, later than \nor same as filing date
4       5                                   Block                                                                     Related to address
5       6                                     Lot                                                                     Related to address
6       7                           Street Number                                                                     Related to address
7       8                    Street Number Suffix                                                                     Related to address
8       9                             Street Name                                                                     Related to address
9      10                      Street Name Suffix                                                                     Related to address
10     11                                    Unit                                                                     Unit of a building
11     12                             Unit suffix                                                            Suffix if any, for the unit
12     13                             Description         Details about purpose of the permit.\n Example: reroofing, bathroom renovation
13     14                          Current Status                                             Current status of the permit application. 
14     15                     Current Status Date                                               Date at which current status was entered
15     16                              Filed Date                                                              Filed date for the permit
16     17                             Issued Date                                                             Issued date for the permit
17     18                          Completed Date  The date on which project was completed, \napplicable if Current Status = “completed”
18     19        First Construction Document Date                                              Date on which construction was documented
19     20                 Structural Notification                                     Notification to meet some legal need, given or not
20     21              Number of Existing Stories  Number of existing stories in the building. \nNot applicable for certain permit types
21     22              Number of Proposed Stories                             Number of proposed stories for the construction/alteration
22     23         Voluntary Soft-Story \nRetrofit                                             Soft story to meet earth quake regulations
23     24                        Fire Only Permit                                                  Fire hazard prevention related permit
24     25                  Permit Expiration Date                                              Expiration date related to issued permit.
25     26                          Estimated Cost                                          Initial estimation of the cost of the project
26     27                            Revised Cost                                          Revised estimation of the cost of the project
27     28                            Existing Use                                                           Existing use of the building
28     29                          Existing Units                                                               Existing number of units
29     30                            Proposed Use                                                           Proposed use of the building
30     31                          Proposed Units                                                               Proposed number of units
31     32                                Plansets         Plan representation indicating the general design\n intent of the foundation..
32     33                         TIDF Compliance                                 TIDF compliant or not, this is a new legal requirement
33     34              Existing Construction Type                    Construction type, existing,as categories \nrepresented numerically
34     35  Existing Construction Type Description              Description of the above, for example, \nwood or other construction types
35     36              Proposed Construction Type                   Construction type, proposed, as categories\n represented numerically
36     37  Proposed Construction Type Description                                                               Description of the above
37     38                             Site Permit                                                                        Permit for site
38     39                     Supervisor District                          Supervisor District to which the building location belongs to
39     40     Neighborhoods - Analysis Boundaries                                 Neighborhood to which the building location belongs to
40     41                                 Zipcode                                                            Zipcode of building address
41     42                                Location                                                  Location in latitude, longitude pair.
42     43                               Record ID                                                           Some ID, not useful for this


```

En esta hoja de cálculo (basada en el conocido conjunto de datos de permisos de construcción de San Francisco), existen **relaciones directas, dependencias jerárquicas e inercia de datos** entre varios atributos. Estas relaciones se pueden aprovechar para imputar o rellenar valores faltantes (*missing values*):

---

### 1. Relaciones Directas 1 a 1 (Código vs. Descripción)

Son atributos donde uno es la representación numérica o clave y el otro es el texto explicativo. Si falta uno, se deduce directamente del otro mediante una tabla de equivalencias:

* **`Permit Type` $\leftrightarrow$ `Permit Type Definition**`: El código del tipo de permiso determina de forma exacta su definición (ejemplo: `1` = *Nueva construcción*, `8` = *Alteraciones*).
* **`Existing Construction Type` $\leftrightarrow$ `Existing Construction Type Description**`: Mapeo directo entre la categoría numérica del tipo de construcción actual y su texto (ejemplo: madera, acero, etc.).
* **`Proposed Construction Type` $\leftrightarrow$ `Proposed Construction Type Description**`: Mapeo idéntico para la estructura propuesta.

---

### 2. Atributos de Ubicación y Geografía (Lookup / Geocodificación)

La ubicación física vincula varios campos en cascada. Si conoces la dirección precisa, puedes rellenar los datos geográficos faltantes buscando otros registros con la misma dirección o bloque:

* **`Block` + `Lot` + `Street Number` + `Street Name` $\leftrightarrow$ `Zipcode`, `Supervisor District`, `Neighborhoods - Analysis Boundaries`, `Location` (Latitud/Longitud)**:
* Si falta el **`Zipcode`**, el **`Supervisor District`**, el **`Neighborhood`** o la **`Location`** de un registro, basta con buscar registros que compartan el mismo `Block` y `Lot` (o la misma calle y número) para copiar sus valores geográficos exactos.



---

### 3. Comparación de Estado Existente vs. Propuesto (Inercia de Cambios)

En la mayoría de las remodelaciones o alteraciones menores, muchas propiedades del edificio no cambian:

* **`Existing Use` $\leftrightarrow$ `Proposed Use**`: Si no se trata de un permiso de cambio de uso (*change of use*), si falta `Proposed Use`, muy probablemente sea idéntico a `Existing Use` (ejemplo: *Single Family Dwelling*).
* **`Existing Units` $\leftrightarrow$ `Proposed Units**`: En proyectos que no amplían viviendas, la cantidad de unidades propuestas suele ser igual a las existentes.
* **`Number of Existing Stories` $\leftrightarrow$ `Number of Proposed Stories**`: Si la descripción no indica adición de niveles, el número de pisos propuestos es igual al existente.
* **`Existing Construction Type` $\leftrightarrow$ `Proposed Construction Type**`: El material de construcción propuesto suele ser igual al existente a menos que se trate de una reconstrucción mayor.

---

### 4. Ciclo de Vida del Permiso y Estado Actual (Flujo Temporal)

El estado de la solicitud impone reglas lógicas sobre las fechas y estados:

* **`Current Status` $\leftrightarrow$ `Completed Date` / `Issued Date` / `Current Status Date**`:
* Si `Current Status` es **"completed"** y falta `Completed Date`, se puede imputar usando `Current Status Date`.
* Si `Current Status` es **"withdrawn"** o **"cancelled"**, las fechas como `Issued Date` o `Completed Date` no aplican o deben permanecer vacías (`N/A`).


* **Secuencia de fechas**: Existe un orden cronológico estricto: `Filed Date` $\le$ `Permit Creation Date` $\le$ `Issued Date` $\le$ `Completed Date`. Si falta alguna de estas fechas intermedias, se puede acotar el rango plausible o estimar mediante la mediana de tiempo de procesamiento para ese `Permit Type`.

---

### 5. Atributos Económicos y de Costos

* **`Estimated Cost` $\leftrightarrow$ `Revised Cost**`:
* Al inicio de un trámite, si falta `Revised Cost`, se suele rellenar de forma preliminar con el `Estimated Cost`.


* **`Estimated Cost` / `Revised Cost` $\leftrightarrow$ `Permit Type` + `Description` + `Proposed Units**`:
* Si falta el costo, se puede imputar estadísticamente (usando la mediana del grupo) agrupando por `Permit Type` y palabras clave en `Description` (por ejemplo, los permisos para *"reroofing"* o *"bathroom renovation"* tienen rangos de costo muy definidos).



---

### 6. Clasificación a través de Texto Libre (`Description`)

El campo de texto libre `Description` contiene detalles valiosos para llenar atributos categóricos faltantes mediante búsqueda de palabras clave:

* **`Description` $\rightarrow$ `Fire Only Permit**`: Si en la descripción aparecen términos como *"sprinkler"*, *"fire alarm"*, *"fire suppression"*, se puede inferir si es un permiso exclusivo de prevención de incendios.
* **`Description` $\rightarrow$ `Voluntary Soft-Story Retrofit**`: Si incluye palabras como *"soft story"* o *"seismic retrofit"*, permite rellenar esta bandera booleana.