# Data

Populated on first run — this directory ships empty on purpose.

```bash
python -m bales.data                # writes both sample CSVs
python scripts/run_experiment.py    # generates them automatically if absent
```

## Files

### `pet_local_prices_sample.csv` — transaction ledger

| Column | Description |
|---|---|
| `Date` | Transaction date |
| `Supplier Name` | Supplier identifier |
| `CAT` | PET grade (Clear, Blue, Green, Mixed) |
| `Price` | Price per kg in **EGP** |
| `Weight` | Gross bale weight (kg) |
| `Deduction` | Fraction deducted for contamination |
| `Accept` | Accepted weight (kg) |
| `Reject` | Rejected weight (kg) |
| `Cost` | Accepted weight × price (EGP) |
| `Invoice` | Amount paid including shipping (EGP) |
| `Carta Series` | Consignment note code |
| `Supply Area` | Collection governorate |
| `Bulking Station Name` | Receiving station |

### `official_exchange_rates_sample.csv` — EGP/USD rates

| Column | Description |
|---|---|
| `Date` | Business days only |
| `Buy` | Official buy rate |
| `Sell` | Official sell rate — used for USD conversion |

## What the generator reproduces

The original ledger was private. The generator recreates its structure and the
market conditions that make the problem interesting:

- **Three real devaluation steps** — March 2022, October 2022, January 2023,
  taking EGP/USD from ~15.7 to ~29.7
- **Grade premiums** — Clear PET at a premium, Mixed at a discount
- **AR(1) price wander** plus annual seasonality over a slow upward USD trend
- **Lighter Friday trading**
- **Data quality issues** — duplicate rows (~0.8%) and missing prices (~1.2%)
- **FX published on business days only**, so weekend trades need the last quote

## Using your own data

Point `bales.data.load()` at a directory containing two CSVs with the schemas
above. Nothing else in the pipeline is coupled to the generator.
