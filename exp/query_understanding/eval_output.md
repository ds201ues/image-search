
### baseline (threshold 0.2)

| type | n | P@10 | R@pool | empty | junk/q | shown |
|---|---|---|---|---|---|---|
| person | 19 | 0.64 | 0.64 | 2 | 0.74 | 4.9 |
| person_context | 29 | 0.37 | 0.57 | 5 | 1.03 | 2.5 |
| scene | 65 | 0.65 | 0.74 | 4 | 0.18 | 12.2 |
| attribute | 19 | 0.62 | 0.92 | 0 | 0.47 | 23.9 |
| place | 23 | 0.64 | 0.37 | 4 | 0.17 | 5.5 |
| event_year | 27 | 0.38 | 0.00 | 2 | 0.07 | 0.2 |
| hinglish | 16 | 0.71 | 0.15 | 5 | 0.00 | 0.5 |
| ALL | 198 | 0.59 | 0.63 | 22 | 0.36 | 7.8 |

### parsed (threshold 0.2)

| type | n | P@10 | R@pool | empty | junk/q | shown |
|---|---|---|---|---|---|---|
| person | 19 | 0.67 | 0.64 | 2 | 0.68 | 4.9 |
| person_context | 29 | 0.31 | 0.53 | 5 | 1.14 | 3.7 |
| scene | 65 | 0.65 | 0.78 | 3 | 0.22 | 12.8 |
| attribute | 19 | 0.62 | 0.92 | 0 | 0.47 | 23.9 |
| place | 23 | 0.61 | 0.52 | 2 | 0.26 | 8.5 |
| event_year | 27 | 0.50 | 0.50 | 1 | 0.07 | 0.4 |
| hinglish | 16 | 0.52 | 0.45 | 2 | 0.31 | 3.4 |
| ALL | 198 | 0.58 | 0.69 | 15 | 0.41 | 8.8 |

### parsed_exp (threshold 0.2)

| type | n | P@10 | R@pool | empty | junk/q | shown |
|---|---|---|---|---|---|---|
| person | 19 | 0.68 | 0.64 | 2 | 0.63 | 4.8 |
| person_context | 29 | 0.31 | 0.53 | 5 | 1.14 | 3.7 |
| scene | 65 | 0.65 | 0.78 | 3 | 0.22 | 12.9 |
| attribute | 19 | 0.62 | 0.92 | 0 | 0.37 | 23.1 |
| place | 23 | 0.61 | 0.51 | 2 | 0.26 | 8.4 |
| event_year | 27 | 0.50 | 0.50 | 1 | 0.07 | 0.4 |
| hinglish | 16 | 0.54 | 0.45 | 2 | 0.25 | 3.1 |
| ALL | 198 | 0.58 | 0.69 | 15 | 0.39 | 8.7 |

### city_boost (threshold 0.2)

| type | n | P@10 | R@pool | empty | junk/q | shown |
|---|---|---|---|---|---|---|
| person | 19 | 0.67 | 0.64 | 2 | 0.68 | 4.9 |
| person_context | 29 | 0.31 | 0.53 | 5 | 1.14 | 3.7 |
| scene | 65 | 0.65 | 0.78 | 3 | 0.22 | 13.0 |
| attribute | 19 | 0.62 | 0.92 | 0 | 0.47 | 23.9 |
| place | 23 | 0.61 | 0.52 | 2 | 0.26 | 8.8 |
| event_year | 27 | 0.50 | 0.50 | 1 | 0.07 | 0.4 |
| hinglish | 16 | 0.52 | 0.45 | 2 | 0.38 | 3.7 |
| ALL | 198 | 0.58 | 0.69 | 15 | 0.42 | 8.9 |

### city_filter (threshold 0.2)

| type | n | P@10 | R@pool | empty | junk/q | shown |
|---|---|---|---|---|---|---|
| person | 19 | 0.67 | 0.64 | 2 | 0.68 | 4.9 |
| person_context | 29 | 0.32 | 0.53 | 5 | 1.03 | 3.6 |
| scene | 65 | 0.66 | 0.76 | 4 | 0.20 | 12.2 |
| attribute | 19 | 0.65 | 0.92 | 0 | 0.37 | 22.9 |
| place | 23 | 0.77 | 0.36 | 6 | 0.00 | 5.8 |
| event_year | 27 | 0.40 | 0.00 | 2 | 0.07 | 0.2 |
| hinglish | 16 | 0.62 | 0.45 | 3 | 0.19 | 3.1 |
| ALL | 198 | 0.60 | 0.65 | 22 | 0.34 | 8.2 |

### Summary at threshold 0.20

| variant | P@10 | R@pool | empty | junk/q | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| baseline | 0.592 | 0.630 | 22 | 0.36 | 489 | 2151 |
| parsed | 0.576 | 0.689 | 15 | 0.41 | 474 | 1852 |
| parsed_exp | 0.578 | 0.688 | 15 | 0.39 | 483 | 1955 |
| city_boost | 0.576 | 0.689 | 15 | 0.42 | 528 | 2087 |
| city_filter | 0.598 | 0.654 | 22 | 0.34 | 460 | 1938 |

### Biggest per-query changes (parsed vs baseline, P@10)

- +1.00 [parsed/scene] `sportsaid athletes with duchess of cambridge 2016` -> `SportsAid athletes with Catherine, Princess of Wales` city='': P@10 0.00 -> 1.00
- +1.00 [city_boost/scene] `sportsaid athletes with duchess of cambridge 2016` -> `SportsAid athletes with Catherine, Princess of Wales` city='': P@10 0.00 -> 1.00
- +1.00 [city_filter/scene] `sportsaid athletes with duchess of cambridge 2016` -> `SportsAid athletes with Catherine, Princess of Wales` city='': P@10 0.00 -> 1.00
- -1.00 [city_filter/place] `chandigarh capitol complex at night` -> `capitol complex at night` city='chandigarh': P@10 1.00 -> 0.00
- +1.00 [parsed/event_year] `old delhi traffic 2023` -> `old traffic` city='delhi': P@10 0.00 -> 1.00
- +1.00 [city_boost/event_year] `old delhi traffic 2023` -> `old traffic` city='delhi': P@10 0.00 -> 1.00
- +0.90 [parsed/hinglish] `actor car se utarte hue` -> `actor getting out of car` city='': P@10 0.00 -> 0.90
- +0.90 [city_boost/hinglish] `actor car se utarte hue` -> `actor getting out of car` city='': P@10 0.00 -> 0.90
- +0.90 [city_filter/hinglish] `actor car se utarte hue` -> `actor getting out of car` city='': P@10 0.00 -> 0.90
- +0.67 [parsed/hinglish] `bollywood actresses stage par` -> `Bollywood actresses on stage` city='': P@10 0.00 -> 0.67
- +0.67 [city_boost/hinglish] `bollywood actresses stage par` -> `Bollywood actresses on stage` city='': P@10 0.00 -> 0.67
- +0.67 [city_filter/hinglish] `bollywood actresses stage par` -> `Bollywood actresses on stage` city='': P@10 0.00 -> 0.67
- +0.57 [parsed/place] `film premiere in mumbai` -> `film premiere` city='mumbai': P@10 0.00 -> 0.57
- +0.57 [city_boost/place] `film premiere in mumbai` -> `film premiere` city='mumbai': P@10 0.00 -> 0.57
- +0.55 [city_filter/attribute] `man in blue shirt at bara imambara` -> `man in blue shirt at Bara Imambara` city='lucknow': P@10 0.45 -> 1.00
- -0.50 [parsed/person_context] `model walking ramp in aslam khan outfit` -> `model walking runway in Aslam Khan outfit` city='': P@10 1.00 -> 0.50
- -0.50 [city_boost/person_context] `model walking ramp in aslam khan outfit` -> `model walking runway in Aslam Khan outfit` city='': P@10 1.00 -> 0.50
- -0.50 [city_filter/person_context] `model walking ramp in aslam khan outfit` -> `model walking runway in Aslam Khan outfit` city='': P@10 1.00 -> 0.50
- -0.50 [city_filter/scene] `security personnel in rain srinagar` -> `security personnel in rain` city='srinagar': P@10 0.50 -> 0.00
- +0.50 [parsed/event_year] `mumbai coastal road construction 2023` -> `coastal road construction` city='mumbai': P@10 0.00 -> 0.50
- +0.50 [city_boost/event_year] `mumbai coastal road construction 2023` -> `coastal road construction` city='mumbai': P@10 0.00 -> 0.50
- +0.50 [parsed/event_year] `srinagar disinfection drive 2020` -> `disinfection drive` city='srinagar': P@10 0.00 -> 0.50
- +0.50 [city_boost/event_year] `srinagar disinfection drive 2020` -> `disinfection drive` city='srinagar': P@10 0.00 -> 0.50
- +0.50 [parsed/scene] `people enjoying first rain bangalore` -> `people enjoying first rain` city='bengaluru': P@10 0.00 -> 0.50
- +0.50 [city_boost/scene] `people enjoying first rain bangalore` -> `people enjoying first rain` city='bengaluru': P@10 0.00 -> 0.50