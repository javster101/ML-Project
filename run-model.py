#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


raw_data = pandas.read_csv('./listings.csv').dropna(subset=['price'])

# Split amenities into features
# Only keep non-unique amenity listings
raw_data['amenities'] = raw_data['amenities'].apply(json.loads)

all_amenities = []
all_amenities_second = []

for amenities_listing in raw_data['amenities']:
    for amenity in amenities_listing:
        if not amenity in all_amenities:
            all_amenities.append(amenity)
        elif amenity in all_amenities and not amenity in all_amenities_second:
            all_amenities_second.append(amenity)

raw_data[all_amenities_second] = [x in raw_data['amenities'] for x in all_amenities_second]
raw_data[all_amenities_second] = raw_data[all_amenities_second].astype(float)
raw_data.drop('amenities', axis=1, inplace=True)

raw_data = raw_data.copy()

print(raw_data)

# Fix percentages
percent_columns = ['host_response_rate', 'host_acceptance_rate']
for percent_column in percent_columns:
    raw_data[percent_column] = raw_data[percent_column].str.rstrip('%').astype(float) / 100

raw_data['price'] = raw_data['price'].str.strip('$').str.replace(',', '').astype(float)

# Fix booleans
boolean_columns = ['host_has_profile_pic', 'host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']
for boolean_column in boolean_columns:
    raw_data[boolean_column] = raw_data[boolean_column].replace({'t': 1.0, 'f': 0.0})

# Strip non-feature rows
features = ['host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_listings_count', 'host_total_listings_count', 
            'host_has_profile_pic', 'host_identity_verified', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
            'minimum_nights', 'maximum_nights', 'has_availability', 'availability_30', 'availability_60', 'availability_90',
            'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
            'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
            'review_scores_value', 'instant_bookable', 'calculated_host_listings_count', 'reviews_per_month']
features.extend(all_amenities_second)

columns = raw_data.columns.tolist()

for column in columns:
    if not column in features and column != 'price':
        raw_data.drop(column, axis=1, inplace=True)

# Convert feature rows to numerical
# Also drop NaNs
raw_data.dropna(inplace=True)

for feature in features:
    if not feature in columns:
        print(f"Feature {feature} is missing in the dataset")
    if not raw_data.dtypes[feature] in ['float']:
        print(f"Feature {feature} is a {raw_data.dtypes[feature]} list, converting")
        raw_data[feature] = raw_data[feature].astype(float)


# Attempt linear regression
feature_set = raw_data[features]
score_set = raw_data['price']

print(feature_set)
print(score_set)

feature_train, feature_test, y_train, y_test = train_test_split(
    feature_set, score_set, test_size=0.30, random_state=42)
print("Randomly split dataset to %d training and %d test samples" % (len(feature_train),len(feature_test)))

model = LinearRegression()
model.fit(feature_train, y_train)


# Compute RMSE on train and test sets
rmse_train = rmse(y_train,model.predict(feature_train))
rmse_test = rmse(y_test,model.predict(feature_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train, rmse_test))

# Attempt neural network
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_set.shape[1], 512),
            nn.PReLU(),
            nn.Linear(512, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class RentalDataset(Dataset):
    def __init__(self, features, scores):
        self.features = features
        self.scores = scores
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        target = torch.tensor(self.scores.iloc[idx], dtype=torch.float)
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float)
        return features, target

model = Network().to(device)
print(model)

dataset = RentalDataset(feature_set, score_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)

# Train model
model.train()
size = len(dataset)

losses = []
for epoch in range(100):
    for (X, y) in dataloader:
        X, y = X.to(device), y.to(device)
    
        optimizer.zero_grad()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
    
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()
    
    losses.append(loss.item())
    print(f"loss: {loss.item():>7f}")

# Test model
num_batches = len(dataloader)
model.eval()
test_loss = 0

preds = []
targets = []
with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()

        preds.extend(pred)
        targets.extend(y)
test_loss /= num_batches

diffs = torch.abs(torch.tensor(preds) - torch.tensor(targets))
squared_diff = (diffs) ** 2
rmse = torch.sqrt(torch.mean(squared_diff))

print(f"Test Error: \n Median: ${(torch.median(diffs)):>0.2f}, RMSE: ${(rmse):>0.2f}, Avg loss: {test_loss:>8f} \n")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
