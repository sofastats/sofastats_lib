from pathlib import Path

from sofastats.conf import main as main_conf

examples_folder = Path(main_conf.__file__).parent.parent.parent / 'sofastats_examples'
print(f"{examples_folder=}")
files_folder = examples_folder / 'files'
sort_orders_yaml_file_path = files_folder / 'sort_orders.yaml'

## CSVs

books_csv_fpath = files_folder / 'books.csv'

education_csv_fpath = files_folder / 'education.csv'
education_with_missing_categories_csv_fpath = files_folder / 'education_with_missing_categories_for_testing.csv'

people_csv_fpath = files_folder / 'people.csv'
people_with_missing_categories_csv_fpath = files_folder / 'people_with_missing_categories_for_testing.csv'

sports_csv_file_path = files_folder / 'sports.csv'
sports_with_missing_categories_csv_file_path = files_folder / 'sports_with_missing_categories_for_testing.csv'

## categories

age_groups_value_sorted = ['20 to <30', '30 to <40', '40 to <50', '50 to <60', '60 to <70', '70 to <80', '80+', '<20', ]
age_groups_custom_sorted = ['<20', '20 to <30', '30 to <40', '40 to <50', '50 to <60', '60 to <70', '70 to <80', '80+', ]

countries_value_sorted = ['Denmark', 'NZ', 'South Korea', 'USA', ]
countries_custom_sorted = ['USA', 'NZ', 'South Korea', 'Denmark', ]

handedness_value_sorted = ['Ambidextrous', 'Left', 'Right', ]
handedness_custom_sorted = ['Right', 'Left', 'Ambidextrous', ]

home_location_types_value_sorted = ['City', 'Rural', 'Town', ]
home_location_types_custom_sorted = ['City', 'Town', 'Rural', ]

sleep_groups_value_sorted = ['7 to <9 hours', '9+ hours', 'Under 7 hours', ]
sleep_groups_custom_sorted = ['Under 7 hours', '7 to <9 hours', '9+ hours', ]

sports_value_sorted = ['Archery', 'Badminton', 'Basketball', ]
sports_custom_sorted = ['Badminton', 'Archery', 'Basketball', ]
