import yagmail

# Yagmail setup (use app password)
sender_email = "my.kot.app@gmail.com"
sender_password = "vdzyjhfnlymwtzby"  # App password for Gmail

# Create yagmail instance
yag = yagmail.SMTP(sender_email, sender_password)

# Email content
subject = f"ALERT"
body = f"""
ALERT NOTIFICATION
"""

# Send email with custom from address
yag.send(
    to="gaurichandar3@gmail.com", 
    subject=subject,
    contents=body,
)