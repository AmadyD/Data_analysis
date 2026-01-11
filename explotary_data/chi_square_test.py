import pandas as pd
from scipy.stats import chi2_contingency

# Create the contingency table
data = [[20, 30],  # Male: [Like, Dislike]
        [25, 25]]  # Female: [Like, Dislike]


# Create a DataFrame for clarity
df = pd.DataFrame(data, columns=["Like", "Dislike"], index=["Male", "Female"])

print (df)

# Perform the Chi-Square Test
chi2, p, dof, expected = chi2_contingency(df)
# Display results
print("Chi-square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)
print("Expected Frequencies:\n", expected)