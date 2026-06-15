// Copy this in a Google Apps Script project to create a Google Form for human evaluation of the PFSMB translations.
// In the repo, the annotation CSV lives in ../outputs/ and the guideline image lives in ../inputs/.
// When copied into Google Apps Script, this script assumes both files are available in the same Drive folder.

const FORM_TITLE = "PFSMB Human Evaluation";
const CSV_FILE_NAME = "annotation_sheet.csv";
const GUIDELINE_IMAGE_FILE_NAME = "pfsmb_guidelines.png";

const GUIDELINES = [
    "1. Normalize incorrect grammar.",
    "2. Normalize incorrect spelling.",
    "3. Preserve word elongation (character repetitions).",
    "4. Preserve non-standard capitalization.",
    "5. Preserve informal abbreviations such as 'gonna', 'u' and 'bro'.",
    "6. Translate informal acronyms such as 'lol', 'brb' and 'idk' to their equivalents in the target language (whenever possible).",
    "7. Translate hashtags and subreddits (while matching the original casing style) only if they have a grammatical function in the sentence. Otherwise, copy them as they are.",
    "8. Copy URLs, usernames, retweet marks (RT) as they are.",
    "9. Copy emojis and emoticons as they are.",
    "10. Copy atypical punctuation.",
    "11. Translate overt profanity without censorship.",
    "12. Translate self-censored profanity with similar self-censorship in the target language."
 ];

function addGuidelineImage(form) {
  const files = DriveApp.getFilesByName(GUIDELINE_IMAGE_FILE_NAME);

  if (!files.hasNext()) {
    throw new Error("Could not find guideline image: " + GUIDELINE_IMAGE_FILE_NAME);
  }

  const imageBlob = files.next().getBlob();

  form.addImageItem()
    .setTitle("Guideline summary")
    .setImage(imageBlob);
}

function createHumanEvalForm() {
  const rows = loadCsvFromDrive(CSV_FILE_NAME);
  const form = FormApp.create(FORM_TITLE);

  form.setDescription(
    "Human evaluation of French-English machine translation of user-generated content (UGC), i.e. social media comments.\n\n" +
    "You will see the original source, a normalised version of the source, a reference translation, and two anonymous system outputs.\n\n" +
    "Note that the normalized source is only to help you understand the meaning, and that the reference might not be perfect.\n\n" +
    "TRIGGER WARNING: some samples may contain vulgar language (swear words).\n\n" +
    "Please judge which output is better overall and which output better follows the UGC translation guidelines.\n\n" +
    "Guidelines:\n" +
    GUIDELINES.join("\n")
  );

  const responseSheet = SpreadsheetApp.create(FORM_TITLE + " Responses");
  form.setDestination(FormApp.DestinationType.SPREADSHEET, responseSheet.getId());

  rows.forEach((row, i) => {
    form.addPageBreakItem()
      .setTitle("Sample " + (i + 1) + " / " + rows.length);

    addGuidelineImage(form);

    form.addSectionHeaderItem()
      .setTitle("Input")
      .setHelpText(
        "Source:\n" +
        (row.source || "") +
        "\n\n" +
        "Normalised source:\n" +
        (row.normed_source || "")
      );

    form.addSectionHeaderItem()
      .setTitle("Reference translation")
      .setHelpText(row.reference || "");

    form.addSectionHeaderItem()
      .setTitle("Translations")
      .setHelpText(
        "Output A:\n" +
        (row.system_a || "") +
        "\n\n" +
        "Output B:\n" +
        (row.system_b || "")
      );

    addChoiceQuestion(
      form,
      "Which translation is better overall?",
      ["A", "B", "Tie", "Cannot judge"]
    );

    addChoiceQuestion(
      form,
      "Which output better follows the UGC guidelines?",
      ["A", "B", "Tie", "Cannot judge"]
    );

    form.addParagraphTextItem()
      .setTitle("Optional comment");
  });

  Logger.log("Edit URL: " + form.getEditUrl());
  Logger.log("Public/respondent URL: " + form.getPublishedUrl());
  Logger.log("Responses sheet: " + responseSheet.getUrl());
}

function addChoiceQuestion(form, title, options) {
  const item = form.addMultipleChoiceItem();
  item.setTitle(title)
    .setChoices(options.map(option => item.createChoice(option)))
    .setRequired(true);
}

function loadCsvFromDrive(fileName) {
  const files = DriveApp.getFilesByName(fileName);

  if (!files.hasNext()) {
    throw new Error("Could not find CSV file in Drive: " + fileName);
  }

  const file = files.next();
  const csvText = file.getBlob().getDataAsString("UTF-8");
  const values = Utilities.parseCsv(csvText);

  const headers = values[0];
  const rows = values.slice(1);

  return rows.map(row => {
    const obj = {};
    headers.forEach((header, i) => {
      obj[header] = row[i] || "";
    });
    return obj;
  });
}
