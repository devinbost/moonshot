
use default_keyspace;

CREATE TABLE family_plans (
    phone_number TEXT,
    family_member_phone_number TEXT,
    name TEXT,
    age INT,
    device TEXT,
    monthly_usage_min INT,
    support_case_ids LIST<UUID>,
    PRIMARY KEY (phone_number, family_member_phone_number)
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5555',
    'John Smith',
    29,
    'iPhone 6',
    680,
    [uuid(), uuid()]
);
INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5559',
    'Michael Smith',
    42,
    'OnePlus 9 Pro',
    800,
    [uuid(), uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5560',
    'Emily Smith',
    19,
    'iPhone SE',
    500,
    [uuid(), uuid(), uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5561',
    'David Smith',
    50,
    'Google Pixel 6',
    700,
    [uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5562',
    'Olivia Smith',
    24,
    'Samsung Galaxy S20',
    600,
    [uuid(), uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5563',
    'William Smith',
    33,
    'iPhone 13',
    750,
    [uuid(), uuid(), uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5564',
    'Ava Smith',
    16,
    'OnePlus Nord',
    450,
    [uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5565',
    'James Smith',
    29,
    'Google Pixel 4a',
    680,
    [uuid(), uuid()]
);

INSERT INTO family_plans (
    phone_number,
    family_member_phone_number,
    name,
    age,
    device,
    monthly_usage_min,
    support_case_ids
) VALUES (
    '555-555-5555',
    '555-555-5566',
    'Sophia Smith',
    22,
    'iPhone XR',
    520,
    [uuid(), uuid(), uuid()]
);

CREATE TABLE customer_support_transcripts (
    phone_number TEXT,
    transcript_id UUID,
    customer_name TEXT,
    issue_type TEXT,
    resolution_status TEXT,
    transcript TEXT,
    interaction_date TIMESTAMP,
    PRIMARY KEY (phone_number, transcript_id)
);
INSERT INTO customer_support_transcripts (
    phone_number,
    transcript_id,
    customer_name,
    issue_type,
    resolution_status,
    transcript,
    interaction_date
) VALUES (
    '555-555-5555',
    uuid(),
    'John Smith',
    'Billing Query',
    'Resolved',
    'Support Agent: Good morning, this is Sarah from customer support. How can I assist you today?\nJohn Smith: Hi, I received my bill today, and it''s higher than usual. Can you check what''s going on?\nSupport Agent: Sure, I see an additional charge for international calls. Did you make any calls overseas last month?\nJohn Smith: Oh, yes, I did. I wasn''t aware of the extra charges.\nSupport Agent: I understand. As a courtesy, I''ll apply a one-time discount to offset some of these charges.\nJohn Smith: That would be great, thank you!\nSupport Agent: You''re welcome! Is there anything else I can assist you with?',
    toTimestamp(now())
);


INSERT INTO customer_support_transcripts (
    phone_number,
    transcript_id,
    customer_name,
    issue_type,
    resolution_status,
    transcript,
    interaction_date
) VALUES (
    '555-555-5555',
    uuid(),
    'John Smith',
    'Network Issue',
    'Pending',
    'Support Agent: Hello, this is Mike from technical support. How may I help you?\nJohn Smith: Hi Mike, I''ve been experiencing poor network coverage at my home for the past week.\nSupport Agent: I’m sorry to hear that. Let me check the network status in your area.\n... [Checking] ...\nSupport Agent: It seems there are some ongoing maintenance works, which might be causing the issue.\nJohn Smith: How long will it take to fix?\nSupport Agent: The team expects to resolve it within 48 hours. We’ll update you via email.\nJohn Smith: Alright, I’ll wait for the update then. Thanks!',
    toTimestamp(now())
);

INSERT INTO customer_support_transcripts (
    phone_number,
    transcript_id,
    customer_name,
    issue_type,
    resolution_status,
    transcript,
    interaction_date
) VALUES (
    '555-555-5555',
    uuid(),
    'John Smith',
    'Device Support',
    'Resolved',
    'Support Agent: Hi, this is Laura from device support. What can I do for you today?\nJohn Smith: Hello, I just bought a new phone for my son and I''m having trouble setting it up.\nSupport Agent: I''d be happy to help with that. Are you having trouble with the physical setup or the software configuration?\nJohn Smith: It''s more about the software part, especially email setup and data transfer from his old phone.\nSupport Agent: No worries, I’ll guide you through it step by step. First, let’s start with your email setup...\n... [Detailed Setup Instructions] ...\nJohn Smith: That worked, thank you! How about transferring his data?\nSupport Agent: Let’s proceed with the data transfer. Do you have his old device with you?\n... [Data Transfer Process] ...\nJohn Smith: All set now, thanks for your help!\nSupport Agent: My pleasure! Feel free to call us if you need further assistance.',
    toTimestamp(now())
);