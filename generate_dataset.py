import pandas as pd
import random
import re

phishing_templates = [
    "Urgent: Your account has been suspended. Click here to verify your identity: http://192.168.1.1/login.html and restore access immediately.",
    "Dear customer, we noticed unusual activity. Please update your password at http://bit.ly/verify-account or your account will be locked.",
    "Your Netflix subscription expires today. Click here to continue watching: http://x9z8.netflix-renewal.com/billing",
    "Congratulations! You've won a $1000 gift card. Claim now at http://free-gift.xyz/winner - limited time offer!",
    "Security Alert: Your bank account requires immediate verification. Login at http://secure-bank-login.com/verify to avoid suspension.",
    "Your PayPal account has been limited. Confirm your identity now: http://paypa1-secure.com/confirm?user=123",
    "UEA University scholarship notification. Verify your details at http://scholarship-uea.edu/claim to receive your award.",
    "Amazon gift card offer: Click here to get your $500 Amazon gift card. http://amaz-on-gift.xyz/claim-now",
    "Your Microsoft account is temporarily locked. Unlock it here: http://account-microsoft-verify.com/unlock",
    "IRS Tax Refund: You are eligible for a $950 refund. Submit your bank details at http://irs-refund-2024.com/claim",
    "LinkedIn: You have 5 new profile viewers. See who viewed your profile: http://linked-in-login.com/viewers",
    "Your email storage is 98% full. Upgrade now to avoid suspension: http://email-upgrade-free.com/storage",
    "Package delivery failed. Reschedule your delivery: http://track-package-ups.com/reschedule",
    "Your Apple ID is locked for security reasons. Verify now: http://apple-id-locked.com/verify",
    "Facebook account compromise detected. Secure your account: http://facebook-secure-login.com/protect",
    "Your Chase Bank account requires immediate attention. Verify your account here: http://chase-bank-secure.com/verify",
    "You have an unclaimed insurance refund. Claim your money here: http://insurance-refund-claim.com/get",
    "Your Instagram account will be deleted in 24 hours. Verify your identity: http://instagram-verify.com/save-account",
    "Credit card payment declined. Update your payment info: http://credit-card-update.com/payment",
    "Your Dropbox account is out of storage. Upgrade now: http://dropbox-free-storage.com/upgrade",
    "University of Cambridge: Your student account is pending. Verify here: http://cambridge-student.com/verify",
    "Gmail Security Alert: Someone tried to access your account. Block access: http://gmail-security-alert.com/block",
    "Your eBay account is temporarily limited. Resolve this now: http://ebay-account-limit.com/resolve",
    "25% discount on all products today only! Shop now: http://discount-offer-24h.com/shop",
    "Your Wells Fargo account needs verification. Click here: http://wellsfargo-verify.com/account",
    "Netflix: Update your payment information. Click here to avoid cancellation: http://netflix-payment-update.com",
    "Your Spotify account premium expires. Renew at 50% off: http://spotify-premium-50off.com/renew",
    "Ticketmaster: You have 2 unread messages about your order. View here: http://ticketmaster-msg.com/view",
    "Your Yahoo account was accessed from a new device. Verify identity: http://yahoo-security-verify.com",
    "Google Docs: Shared document access required. Click to view: http://google-docs-shared.com/view",
    "Your Hulu subscription auto-renew failed. Update payment: http://hulu-payment-update.com/billing",
    "Bank of America: Verify your recent transaction. Click here: http://bankofamerica-transaction.com/verify",
    "Your Adobe account password expires in 24 hours. Reset now: http://adobe-password-reset.com",
    "Uber: You have a free ride credit. Claim here: http://uber-free-ride.com/claim",
    "Your iCloud storage is full. Upgrade for free: http://icloud-storage-free.com/upgrade",
    "American Express: Your card is blocked. Verify your identity: http://amex-card-blocked.com/verify",
    "Your Discord account is under attack. Secure it now: http://discord-secure-account.com",
    "Zoom: Meeting link expired. Renew your plan: http://zoom-meeting-renew.com/payment",
    "Your WhatsApp account is banned. Appeal here: http://whatsapp-ban-appeal.com",
    "Payoneer: $2000 received in your account. Withdraw now: http://payoneer-withdraw.com/cash"
]

safe_templates = [
    "Hi John, just wanted to follow up on our meeting scheduled for tomorrow at 10 AM in the conference room.",
    "Thanks for your email. I've attached the quarterly report you requested. Let me know if you have any questions.",
    "Dear team, please review the attached project proposal and provide your feedback by end of this week.",
    "Happy birthday! Hope you have a wonderful day. Let's catch up over coffee next week.",
    "The meeting has been rescheduled to Friday 3 PM. Please update your calendars accordingly.",
    "I've reviewed the document you sent. The changes look good. We can proceed with the next phase.",
    "Don't forget about the team lunch tomorrow at noon at the Italian restaurant downtown.",
    "Your appointment is confirmed for next Monday at 2 PM with Dr. Smith at City Medical Center.",
    "Thank you for the interview opportunity. I look forward to hearing from you soon.",
    "The package was delivered to your front door. Tracking number: 1Z9999W9999.",
    "Your flight check-in is now available. You can check in 24 hours before departure.",
    "Please find the attached invoice for the services provided in March.",
    "The weather this weekend looks great for a hike. Want to join us on Saturday?",
    "Your order has been shipped and will arrive in 3-5 business days.",
    "Let's schedule a call to discuss the new project requirements.",
    "I've submitted my expense report for the conference trip.",
    "The children's soccer practice has been moved to Thursday this week.",
    "Your subscription has been renewed successfully for another year.",
    "Looking forward to working with you on the upcoming project.",
    "The presentation went well. Great job everyone on the team.",
    "Your library book is due next Tuesday. Please return it on time.",
    "Don't forget to submit your vacation request by Friday.",
    "The new office location is at 123 Main Street, Suite 400.",
    "Your car service is scheduled for next Wednesday at 8 AM.",
    "Please review the attached contract before our meeting.",
    "Your grocery order has been confirmed and will be delivered today.",
    "The course syllabus has been posted to the learning portal.",
    "Your ticket to the concert has been confirmed. Enjoy the show!",
    "Let me know if you need help with the database migration task.",
    "The quarterly review meeting is scheduled for next Thursday.",
    "Your hotel reservation has been confirmed for your upcoming trip.",
    "Please join the webinar on digital marketing next Tuesday.",
    "Your renovation project has been approved by the homeowners association.",
    "The student orientation is next Monday at 9 AM in the auditorium.",
    "Your pension statement is now available in your online account.",
    "The annual company picnic is scheduled for June 15th.",
    "Your flight has been upgraded to business class. Enjoy your trip!",
    "Please attend the compliance training session next week.",
    "Your water bill is due on the 15th of this month.",
    "The new textbooks have arrived and are available in the bookstore."
]

random.seed(42)

records = []

for _ in range(500):
    txt = random.choice(phishing_templates)
    variation = txt.replace("http://", "http://")
    if random.random() > 0.7:
        words = variation.split()
        if len(words) > 5:
            idx = random.randint(2, min(5, len(words) - 1))
            words.insert(idx, "URGENT:")
            variation = " ".join(words)
    if random.random() > 0.6:
        variation += " Act now!"
    records.append({"email_text": variation, "label": 1})

for _ in range(500):
    txt = random.choice(safe_templates)
    if random.random() > 0.7:
        txt = txt + " Best regards."
    if random.random() > 0.8:
        txt = txt + " Please let me know if you have any questions."
    records.append({"email_text": txt, "label": 0})

random.shuffle(records)

df = pd.DataFrame(records)
df.to_csv("dataset/phishing_emails.csv", index=False)
print(f"Dataset created with {len(df)} records")
print(f"Phishing: {sum(df['label'] == 1)}")
print(f"Safe: {sum(df['label'] == 0)}")
