import test from 'node:test';
import assert from 'node:assert/strict';
import {
  clampGenerationSettings,
  clampPersonalityConfig,
  computeLossDomain,
  formatLearningRate,
  formatParamCount,
  formatRegisterLabel,
  formatStanceLabel,
  formatTrainingStageLabel,
  formatTrainingStatusLabel,
  isProductionChatReady,
  MEMORY_CATEGORIES,
} from './stanState';

test('formatTrainingStageLabel and formatLearningRate render accurate Phase 2 values', () => {
  assert.equal(formatTrainingStageLabel('pretraining'), 'Stage 1: English Pretraining');
  assert.equal(formatTrainingStageLabel('instruction_tuning'), 'Stage 2: Instruction Tuning (SFT)');
  assert.equal(formatTrainingStageLabel('smoke_test'), 'Smoke Test (Synthetic Verification)');
  assert.equal(formatLearningRate(3e-4), '3.00e-4');
  assert.equal(formatLearningRate(null), '—');
});

test('isProductionChatReady strictly gates untrained and smoke_test states', () => {
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: false,
      tokenizer_loaded: false,
      training_status: 'untrained',
    }),
    false,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: true,
      training_status: 'smoke_test',
    }),
    false,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: false,
      training_status: 'trained',
    }),
    false,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: true,
      training_status: 'trained',
    }),
    true,
  );
  assert.equal(
    isProductionChatReady({
      checkpoint_loaded: true,
      tokenizer_loaded: true,
      training_status: 'instruction-tuned',
    }),
    true,
  );
});

test('clampGenerationSettings bounds all parameters cleanly', () => {
  const clamped = clampGenerationSettings({
    temperature: 5.5,
    top_k: -10,
    top_p: 0,
    max_new_tokens: 9999,
    repetition_penalty: 0.4,
    seed: -3,
    apiBaseUrl: 'http://127.0.0.1:8001/',
  });
  assert.equal(clamped.temperature, 2);
  assert.equal(clamped.top_k, 0);
  assert.equal(clamped.top_p, 0.05);
  assert.equal(clamped.max_new_tokens, 512);
  assert.equal(clamped.repetition_penalty, 1);
  assert.equal(clamped.seed, null);
  assert.equal(clamped.apiBaseUrl, 'http://127.0.0.1:8001');
});

test('formatParamCount and status labels render accurate human-readable values', () => {
  assert.equal(formatParamCount(51658240), '51,658,240 (51.66M)');
  assert.equal(formatTrainingStatusLabel('untrained'), 'Untrained (No Checkpoint)');
  assert.equal(formatTrainingStatusLabel('instruction-tuned'), 'Instruction-Tuned');
});

test('computeLossDomain extracts valid curves and handles empty histories', () => {
  const empty = computeLossDomain([], 'train_loss');
  assert.equal(empty.points.length, 0);

  const domain = computeLossDomain(
    [
      { step: 10, train_loss: 4.0, val_loss: null, learning_rate: 3e-4 },
      { step: 20, train_loss: 2.5, val_loss: 2.8, learning_rate: 2e-4 },
    ],
    'train_loss',
  );
  assert.equal(domain.points.length, 2);
  assert.equal(domain.minStep, 10);
  assert.equal(domain.maxStep, 20);
  assert.ok(domain.minVal < 2.5);
  assert.ok(domain.maxVal > 4.0);
});

test('MEMORY_CATEGORIES and clampPersonalityConfig preserve all 7 stores and valid bounds', () => {
  assert.equal(MEMORY_CATEGORIES.length, 7);
  assert.ok(MEMORY_CATEGORIES.includes('identity_personality'));
  assert.ok(MEMORY_CATEGORIES.includes('people_relationships'));
  assert.ok(MEMORY_CATEGORIES.includes('user_facts'));
  assert.ok(MEMORY_CATEGORIES.includes('conversation_history'));
  assert.ok(MEMORY_CATEGORIES.includes('project_knowledge'));
  assert.ok(MEMORY_CATEGORIES.includes('learned_preferences'));
  assert.ok(MEMORY_CATEGORIES.includes('decisions_experiences'));

  const clamped = clampPersonalityConfig({
    directness: 2.4,
    independence: -0.5,
    humor: 0.65,
    allow_disagreement: false,
  });
  assert.equal(clamped.directness, 1);
  assert.equal(clamped.independence, 0);
  assert.equal(clamped.humor, 0.65);
  assert.equal(clamped.allow_disagreement, false);
  assert.equal(formatStanceLabel('reassess'), 'Self-Correction');
  assert.equal(formatRegisterLabel('blunt'), 'Blunt');
});
