# Multi-vehicle validation suite

יצרתי לך 5 תרחישים חדשים לבדיקה של הסימולטור כאשר יש כמה רכבים במהירויות שונות.
כולם מוגדרים עם `traffic_level: "none"` כדי לצמצם auto-spawn של רכבים נוספים דרך מנגנון צפיפות התנועה.

## הקבצים
- `scenario_02_three_lanes_three_speeds.json` — 3 רכבים, 3 נתיבים, מהירויות שונות, DAS בלבד.
- `scenario_03_counterflow_mixed_speeds.json` — תנועה דו-כיוונית עם שני חיישני DAS.
- `scenario_04_bus_plus_cars_mixed_mass_speed.json` — אוטובוס ושתי מכוניות לבדיקת מסה מול מהירות.
- `scenario_05_multi_target_fusion.json` — 4 רכבים ל-multi-target fusion עם camera + DAS + Kalman.
- `scenario_06_close_pass_challenge.json` — תרחיש מאתגר של closing-gap כדי לבדוק היכן המעקב נשבר.

## סדר ריצה מומלץ
1. scenario_02 — לבדוק קודם slope, SNR ו-RMSE_fiber במצב נקי.
2. scenario_04 — לבדוק מסה לעומת מהירות.
3. scenario_03 — לבדוק כיוונים מנוגדים.
4. scenario_05 — לבדוק fusion מרובה מטרות.
5. scenario_06 — לבדוק גבולות/כשל.

## חשוב לפני הרצה
בנוסף ל-`traffic_level: "none"`, ודאי שב-GUI כבוי גם:
- Random Populate / continuous spawning
- כל אפשרות של auto-spawn בזמן ריצה

לפי הקוד שלך, יש שני מנגנונים שונים שעלולים להוסיף רכבים:
1. בקרת צפיפות תנועה לפי `traffic_level`
2. continuous spawning מתוך חלון ה-Random Populate

## מה למדוד בכל תרחיש
- slope לכל רכב בנפרד
- RMSE לאורך הסיב (`RMSE_fiber`) ולא רק RMSE 2D
- SNR ממוצע וגם SNR בנקודת ייחוס מוקדמת כשהמרחקים דומים
- MAE / RMSE של Kalman לכל רכב בנפרד בתרחישי fusion
- האם יש confusion / identity switch כאשר רכבים מתקרבים