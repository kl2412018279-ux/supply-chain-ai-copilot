# Power BI management dashboard

The platform writes report-ready tables (all starting with `bi_`) into the same
PostgreSQL database it runs on (Supabase). Power BI connects to that database, so the
management report always shows the same numbers as the platform.

> Power BI Desktop runs on **Windows** only and is free to download from the Microsoft
> Store. Publishing a report to share with others needs a Power BI Pro licence; building
> and using it on your own computer does not.

## 1. The tables

| Table | One row per | Use it for |
|---|---|---|
| `bi_jobs` | completed job | Revenue, cost, margin, margin %, each leak in RM (`leak_waiting_rm`, `leak_demurrage_rm`, `leak_late_penalty_rm`, `leak_damage_rm`, `leak_diesel_rm`), `loss_making` |
| `bi_free_time` | live import container (today's snapshot) | Demurrage / detention at risk: `days_left`, `urgency`, `accrued_rm`, `projected_rm`, `action` |
| `bi_deliveries` | delivery | On-time and damage rates by haulier and month |
| `bi_volumes` | day × commodity | TEU trend |
| `bi_carriers` | carrier | Lookup: carrier name, mode, price factor |
| `bi_destinations` | destination | Lookup: name, distance, latitude / longitude for maps |
| `bi_refresh` | (one row) | When the feed was last rebuilt |

The tables are rebuilt every time the app starts. To rebuild them by hand:

```bash
DATABASE_URL="postgresql://..." python -m src.bi.export
python -m src.bi.export --csv powerbi_data/   # or CSV files, no database needed
```

## 2. Give Power BI a read-only login (recommended)

Don't put your main database password into Power BI. In Supabase, open **SQL Editor**,
paste the contents of `docs/powerbi/readonly_role.sql`, change the password, and press
**Run**. It creates a `powerbi_reader` login that can only read the seven `bi_` tables.

The `bi_` tables have Row Level Security turned on, so Supabase's public API keys
cannot read them; the script adds a read policy for `powerbi_reader` only. The app
rebuilds the tables on every start and adds the grant and policy back itself, so you
run the script once.

## 3. Connect Power BI Desktop

1. **Home → Get data → More… → PostgreSQL database → Connect.**
2. **Server**: the host and port from Supabase's **Connect → Session pooler** string,
   e.g. `aws-0-ap-southeast-1.pooler.supabase.com:5432`.
   **Database**: `postgres`. **Data connectivity mode**: *Import*.
3. Choose **Database** credentials. **User name**: `powerbi_reader.<your-project-ref>`
   (the pooler needs the project reference after a dot, the same as in your
   connection string). **Password**: the one you set in step 2.
4. Tick the seven `public.bi_…` tables and press **Load**.

## 4. Relationships (Model view)

Drag to connect, all *many-to-one*, single direction:

- `bi_jobs[haulier]` → `bi_carriers[haulier]`
- `bi_jobs[destination]` → `bi_destinations[destination]`
- `bi_deliveries[haulier]` → `bi_carriers[haulier]`
- `bi_deliveries[destination]` → `bi_destinations[destination]`
- `bi_free_time[destination]` → `bi_destinations[destination]`

Optionally add a date table (`Modeling → New table`:
`Calendar = CALENDAR(DATE(2024,1,1), DATE(2027,12,31))`) and relate it to
`bi_jobs[date]`, `bi_deliveries[date]` and `bi_volumes[date]`.

## 5. Measures (DAX)

Create these with **Modeling → New measure** on `bi_jobs` (and the last three on their
tables):

```DAX
Revenue = SUM ( bi_jobs[revenue_rm] )
Gross margin = SUM ( bi_jobs[margin_rm] )
Gross margin % = DIVIDE ( [Gross margin], [Revenue] )
Quoted margin % = DIVIDE ( SUM ( bi_jobs[planned_margin_rm] ), [Revenue] )
Margin gap (pts) = ( [Gross margin %] - [Quoted margin %] ) * 100
Leakage = SUM ( bi_jobs[leakage_rm] )
Leakage % of revenue = DIVIDE ( [Leakage], [Revenue] )
Loss-making jobs = CALCULATE ( COUNTROWS ( bi_jobs ), bi_jobs[loss_making] = TRUE () )

On-time % = DIVIDE (
    CALCULATE ( COUNTROWS ( bi_deliveries ), bi_deliveries[on_time] = TRUE () ),
    COUNTROWS ( bi_deliveries ) )
Containers past free time = CALCULATE ( COUNTROWS ( bi_free_time ), bi_free_time[days_left] < 0 )
Charges at planned move = SUM ( bi_free_time[projected_rm] )
```

Format the RM measures as currency with the `RM` symbol and the % measures as
percentages (Measure tools ribbon).

## 6. Suggested report (3 pages)

1. **Overview**: cards for Revenue, Gross margin %, Margin gap, Leakage, On-time %,
   Containers past free time; a line chart of Gross margin % by `bi_jobs[month]`.
2. **Margin leakage**: stacked bar of the five leak columns by customer; table of
   loss-making jobs; map of `bi_destinations` sized by Leakage.
3. **Free time & carriers**: table of `bi_free_time` sorted by `days_left` with
   conditional formatting on `urgency`; on-time % and damage rate by carrier.

To match the platform's colours, import the theme **View → Themes → Browse for themes**
→ `docs/powerbi/muara-theme.json`.

## 7. Keeping it fresh

In Power BI Desktop press **Refresh**. If you publish to the Power BI Service, set a
scheduled refresh. The service reaches Supabase over the internet, so no gateway is
needed.
