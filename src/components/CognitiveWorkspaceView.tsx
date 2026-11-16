import { useCallback, useEffect, useState, type FormEvent } from 'react';
import { Pencil, Pin, Plus, Search, Trash2, X } from 'lucide-react';
import {
  createMemory,
  deleteMemory,
  fetchMemories,
  searchMemories,
  updateIdentity,
  updateMemory,
} from '../api/stanClient';
import type {
  IdentityProfile,
  MemoryCategory,
  MemoryListResponse,
  MemoryRecord,
  PersonalityConfig,
} from '../types/stan';
import {
  clampPersonalityConfig,
  DEFAULT_PERSONALITY_CONFIG,
  MEMORY_CATEGORIES,
  MEMORY_CATEGORY_META,
} from '../utils/stanState';

interface CognitiveWorkspaceViewProps {
  apiBaseUrl: string;
  identity: IdentityProfile | null;
  onIdentityUpdated: (next: IdentityProfile) => void;
}

type SubTab = 'memory' | 'identity';

export function CognitiveWorkspaceView({
  apiBaseUrl,
  identity,
  onIdentityUpdated,
}: CognitiveWorkspaceViewProps) {
  const [subTab, setSubTab] = useState<SubTab>('memory');

  // Memory state
  const [selectedCategory, setSelectedCategory] = useState<'all' | MemoryCategory>('all');
  const [memoryData, setMemoryData] = useState<MemoryListResponse | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<MemoryRecord[] | null>(null);
  const [memLoading, setMemLoading] = useState(false);
  const [memError, setMemError] = useState<string | null>(null);

  // Add memory form
  const [showAddForm, setShowAddForm] = useState(false);
  const [newCategory, setNewCategory] = useState<MemoryCategory>('user_facts');
  const [newSubject, setNewSubject] = useState('');
  const [newContent, setNewContent] = useState('');
  const [newConfidence, setNewConfidence] = useState(0.95);
  const [newPinned, setNewPinned] = useState(false);

  // Edit / delete memory inline states
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editSubject, setEditSubject] = useState('');
  const [editContent, setEditContent] = useState('');
  const [editConfidence, setEditConfidence] = useState(0.95);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  // Identity & Personality state
  const [personality, setPersonality] = useState<PersonalityConfig>(
    identity?.personality ?? DEFAULT_PERSONALITY_CONFIG,
  );
  const [principlesText, setPrinciplesText] = useState<string>(
    identity?.core_principles.join('\n') ?? '',
  );
  const [identitySaving, setIdentitySaving] = useState(false);
  const [identityStatus, setIdentityStatus] = useState<string | null>(null);
  const [identityError, setIdentityError] = useState<string | null>(null);

  useEffect(() => {
    if (identity) {
      setPersonality(clampPersonalityConfig(identity.personality));
      setPrinciplesText(identity.core_principles.join('\n'));
    }
  }, [identity]);

  const loadMemories = useCallback(async () => {
    setMemLoading(true);
    setMemError(null);
    try {
      const cat = selectedCategory === 'all' ? undefined : selectedCategory;
      const data = await fetchMemories(apiBaseUrl, cat, 150);
      setMemoryData(data);
    } catch (err) {
      setMemError(err instanceof Error ? err.message : 'Failed to load persistent memory store.');
    } finally {
      setMemLoading(false);
    }
  }, [apiBaseUrl, selectedCategory]);

  useEffect(() => {
    void loadMemories();
  }, [loadMemories]);

  async function handleSearchSubmit(e: FormEvent) {
    e.preventDefault();
    const q = searchQuery.trim();
    if (!q) {
      setSearchResults(null);
      return;
    }
    setMemLoading(true);
    setMemError(null);
    try {
      const cats = selectedCategory === 'all' ? undefined : [selectedCategory];
      const res = await searchMemories(apiBaseUrl, q, cats, 20);
      setSearchResults(res.records);
    } catch (err) {
      setMemError(err instanceof Error ? err.message : 'Memory search failed.');
    } finally {
      setMemLoading(false);
    }
  }

  function handleResetSearch() {
    setSearchQuery('');
    setSearchResults(null);
  }

  async function handleCreateMemory(e: FormEvent) {
    e.preventDefault();
    if (!newSubject.trim() || !newContent.trim()) return;
    setMemError(null);
    try {
      await createMemory(apiBaseUrl, {
        category: newCategory,
        subject: newSubject.trim(),
        content: newContent.trim(),
        source: 'owner_entry',
        confidence: newConfidence,
        pinned: newPinned,
      });
      setNewSubject('');
      setNewContent('');
      setNewConfidence(0.95);
      setNewPinned(false);
      setShowAddForm(false);
      setSearchResults(null);
      await loadMemories();
    } catch (err) {
      setMemError(err instanceof Error ? err.message : 'Failed to store memory record.');
    }
  }

  function startEditing(rec: MemoryRecord) {
    setEditingId(rec.id);
    setEditSubject(rec.subject);
    setEditContent(rec.content);
    setEditConfidence(rec.confidence);
    setConfirmDeleteId(null);
  }

  async function handleSaveEdit(recordId: string) {
    if (!editSubject.trim() || !editContent.trim()) return;
    setMemError(null);
    try {
      await updateMemory(apiBaseUrl, recordId, {
        subject: editSubject.trim(),
        content: editContent.trim(),
        confidence: editConfidence,
      });
      setEditingId(null);
      setSearchResults(null);
      await loadMemories();
    } catch (err) {
      setMemError(err instanceof Error ? err.message : 'Failed to update memory record.');
    }
  }

  async function handleTogglePin(rec: MemoryRecord) {
    setMemError(null);
    try {
      await updateMemory(apiBaseUrl, rec.id, { pinned: !rec.pinned });
      await loadMemories();
    } catch (err) {
      setMemError(err instanceof Error ? err.message : 'Failed to update pin state.');
    }
  }

  async function handleDeleteRecord(recordId: string) {
    setMemError(null);
    try {
      await deleteMemory(apiBaseUrl, recordId);
      setConfirmDeleteId(null);
      if (searchResults) {
        setSearchResults((prev) => (prev ? prev.filter((r) => r.id !== recordId) : null));
      }
      await loadMemories();
    } catch (err) {
      setMemError(err instanceof Error ? err.message : 'Failed to delete memory record.');
    }
  }

  async function handleSaveIdentity(e: FormEvent) {
    e.preventDefault();
    setIdentitySaving(true);
    setIdentityStatus(null);
    setIdentityError(null);
    try {
      const cleanedPrinciples = principlesText
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
      const updated = await updateIdentity(apiBaseUrl, {
        personality: clampPersonalityConfig(personality),
        core_principles: cleanedPrinciples,
      });
      onIdentityUpdated(updated);
      setIdentityStatus('Saved to persistent identity configuration.');
    } catch (err) {
      setIdentityError(err instanceof Error ? err.message : 'Failed to save identity profile.');
    } finally {
      setIdentitySaving(false);
    }
  }

  const displayedRecords = searchResults ?? memoryData?.records ?? [];
  const totalCount = memoryData
    ? Object.values(memoryData.counts).reduce((sum, n) => sum + n, 0)
    : 0;

  return (
    <section aria-labelledby="cognitive-heading" className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between border-b border-slate-200 pb-5">
        <div>
          <h1 id="cognitive-heading" className="font-display text-2xl font-bold tracking-tight text-slate-900">
            Memory &amp; Identity Architecture
          </h1>
          <p className="mt-1.5 text-sm text-slate-600">
            Persistent 7-category SQLite memory store and owner-configurable personality distinct from Transformer weights.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setSubTab('memory')}
            aria-pressed={subTab === 'memory'}
            className={`min-h-[40px] rounded-lg px-4 py-2 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              subTab === 'memory'
                ? 'bg-blue-700 text-white'
                : 'border border-slate-300 bg-white text-slate-800 hover:bg-slate-50'
            }`}
          >
            7-Store Persistent Memory
          </button>
          <button
            type="button"
            onClick={() => setSubTab('identity')}
            aria-pressed={subTab === 'identity'}
            className={`min-h-[40px] rounded-lg px-4 py-2 text-xs font-semibold transition-colors whitespace-nowrap shrink-0 ${
              subTab === 'identity'
                ? 'bg-blue-700 text-white'
                : 'border border-slate-300 bg-white text-slate-800 hover:bg-slate-50'
            }`}
          >
            Identity &amp; Personality
          </button>
        </div>
      </div>

      {subTab === 'memory' ? (
        <div className="space-y-6">
          {/* Category Filters & Counts */}
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <button
              type="button"
              onClick={() => {
                setSelectedCategory('all');
                setSearchResults(null);
              }}
              className={`rounded-xl border p-3.5 text-left transition-colors ${
                selectedCategory === 'all'
                  ? 'border-blue-700 bg-blue-50/70'
                  : 'border-slate-200 bg-white hover:bg-slate-50'
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-xs font-semibold text-slate-900">All Memory Stores</span>
                <span className="font-mono text-sm font-bold text-slate-900 tabular-nums">
                  {totalCount}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-600">Unified durable record view</p>
            </button>

            {MEMORY_CATEGORIES.map((cat) => {
              const meta = MEMORY_CATEGORY_META[cat];
              const count = memoryData?.counts?.[cat] ?? 0;
              const active = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  type="button"
                  onClick={() => {
                    setSelectedCategory(cat);
                    setSearchResults(null);
                  }}
                  className={`rounded-xl border p-3.5 text-left transition-colors ${
                    active
                      ? 'border-blue-700 bg-blue-50/70'
                      : 'border-slate-200 bg-white hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-xs font-semibold text-slate-900">{meta.label}</span>
                    <span className="font-mono text-sm font-bold text-slate-900 tabular-nums">
                      {count}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-600 line-clamp-1">{meta.description}</p>
                </button>
              );
            })}
          </div>

          {/* Search + Add Memory Toolbar */}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <form onSubmit={handleSearchSubmit} className="flex flex-1 items-center gap-2">
              <div className="relative flex-1">
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
                  aria-hidden="true"
                />
                <input
                  type="search"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search durable memories by subject or content..."
                  aria-label="Search memories"
                  className="w-full rounded-lg border border-slate-300 bg-white pl-9 pr-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-700 focus:outline-none"
                />
              </div>
              <button
                type="submit"
                disabled={memLoading}
                className="min-h-[40px] rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-900 hover:bg-slate-50 whitespace-nowrap shrink-0"
              >
                Search
              </button>
              {searchResults !== null && (
                <button
                  type="button"
                  onClick={handleResetSearch}
                  className="min-h-[40px] rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 whitespace-nowrap shrink-0"
                >
                  Reset
                </button>
              )}
            </form>

            <button
              type="button"
              onClick={() => setShowAddForm((v) => !v)}
              aria-expanded={showAddForm}
              className="inline-flex min-h-[40px] items-center gap-2 rounded-lg bg-blue-700 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-800 transition-colors whitespace-nowrap shrink-0"
            >
              {showAddForm ? (
                <>
                  <X className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>Close Form</span>
                </>
              ) : (
                <>
                  <Plus className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>Add Memory Record</span>
                </>
              )}
            </button>
          </div>

          {showAddForm && (
            <form
              onSubmit={handleCreateMemory}
              className="rounded-xl border border-slate-200 bg-white p-5 space-y-4"
            >
              <h2 className="font-display text-base font-bold text-slate-900">
                Record Verified Fact, Preference, or Decision
              </h2>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div>
                  <label htmlFor="new-mem-cat" className="block text-xs font-semibold text-slate-700">
                    Memory Store
                  </label>
                  <select
                    id="new-mem-cat"
                    value={newCategory}
                    onChange={(e) => setNewCategory(e.target.value as MemoryCategory)}
                    className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900"
                  >
                    {MEMORY_CATEGORIES.map((c) => (
                      <option key={c} value={c}>
                        {MEMORY_CATEGORY_META[c].label}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label htmlFor="new-mem-subj" className="block text-xs font-semibold text-slate-700">
                    Subject
                  </label>
                  <input
                    id="new-mem-subj"
                    type="text"
                    required
                    value={newSubject}
                    onChange={(e) => setNewSubject(e.target.value)}
                    placeholder="e.g., Response style preference"
                    className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900"
                  />
                </div>
                <div>
                  <label htmlFor="new-mem-conf" className="block text-xs font-semibold text-slate-700">
                    Confidence ({newConfidence.toFixed(2)})
                  </label>
                  <input
                    id="new-mem-conf"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={newConfidence}
                    onChange={(e) => setNewConfidence(Number(e.target.value))}
                    className="mt-3 w-full accent-blue-700"
                  />
                </div>
              </div>
              <div>
                <label htmlFor="new-mem-content" className="block text-xs font-semibold text-slate-700">
                  Memory Content
                </label>
                <textarea
                  id="new-mem-content"
                  rows={3}
                  required
                  value={newContent}
                  onChange={(e) => setNewContent(e.target.value)}
                  placeholder="Enter exact factual content without fabrication..."
                  className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900"
                />
              </div>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <label className="inline-flex items-center gap-2 text-xs font-medium text-slate-700">
                  <input
                    type="checkbox"
                    checked={newPinned}
                    onChange={(e) => setNewPinned(e.target.checked)}
                    className="h-4 w-4 accent-blue-700"
                  />
                  <span>Pin record for priority reasoning retrieval</span>
                </label>
                <button
                  type="submit"
                  className="min-h-[40px] rounded-lg bg-blue-700 px-5 py-2 text-xs font-semibold text-white hover:bg-blue-800"
                >
                  Save Memory Record
                </button>
              </div>
            </form>
          )}

          {memError && (
            <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-xs font-medium text-red-900">
              {memError}
            </div>
          )}

          {/* Records Ledger */}
          <div className="rounded-xl border border-slate-200 bg-white divide-y divide-slate-200">
            {displayedRecords.length === 0 ? (
              <div className="p-8 text-center text-sm text-slate-600">
                {memLoading
                  ? 'Loading persistent memory records...'
                  : 'No memory records match this filter.'}
              </div>
            ) : (
              displayedRecords.map((rec) => {
                const catLabel =
                  MEMORY_CATEGORY_META[rec.category as MemoryCategory]?.label ?? rec.category;
                const isEditing = editingId === rec.id;
                const isConfirmingDelete = confirmDeleteId === rec.id;

                return (
                  <article key={rec.id} className="p-5 space-y-2.5">
                    {isEditing ? (
                      <div className="space-y-3">
                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                          <div>
                            <label className="block text-xs font-semibold text-slate-700">Subject</label>
                            <input
                              type="text"
                              value={editSubject}
                              onChange={(e) => setEditSubject(e.target.value)}
                              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-semibold text-slate-700">
                              Confidence ({editConfidence.toFixed(2)})
                            </label>
                            <input
                              type="range"
                              min={0}
                              max={1}
                              step={0.05}
                              value={editConfidence}
                              onChange={(e) => setEditConfidence(Number(e.target.value))}
                              className="mt-2.5 w-full accent-blue-700"
                            />
                          </div>
                        </div>
                        <div>
                          <label className="block text-xs font-semibold text-slate-700">Content</label>
                          <textarea
                            rows={2}
                            value={editContent}
                            onChange={(e) => setEditContent(e.target.value)}
                            className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm"
                          />
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => void handleSaveEdit(rec.id)}
                            className="min-h-[36px] rounded-lg bg-blue-700 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-800"
                          >
                            Save Correction
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingId(null)}
                            className="min-h-[36px] rounded-lg border border-slate-300 bg-white px-4 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                          <div>
                            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-slate-500">
                              <span className="font-semibold text-sm text-slate-900">
                                {rec.subject}
                              </span>
                              <span aria-hidden="true">·</span>
                              <span className="font-medium text-blue-700">{catLabel}</span>
                              <span aria-hidden="true">·</span>
                              <span className="font-mono">src={rec.source}</span>
                              <span aria-hidden="true">·</span>
                              <span className="font-mono tabular-nums">
                                conf={rec.confidence.toFixed(2)}
                              </span>
                              {rec.pinned && (
                                <>
                                  <span aria-hidden="true">·</span>
                                  <span className="font-semibold text-slate-900">Pinned</span>
                                </>
                              )}
                            </div>
                          </div>

                          <div className="flex items-center gap-1.5 shrink-0">
                            <button
                              type="button"
                              onClick={() => void handleTogglePin(rec)}
                              title={rec.pinned ? 'Unpin memory' : 'Pin memory'}
                              className="inline-flex min-h-[34px] items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                            >
                              <Pin className="h-3 w-3" aria-hidden="true" />
                              <span>{rec.pinned ? 'Unpin' : 'Pin'}</span>
                            </button>
                            <button
                              type="button"
                              onClick={() => startEditing(rec)}
                              className="inline-flex min-h-[34px] items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
                            >
                              <Pencil className="h-3 w-3" aria-hidden="true" />
                              <span>Edit</span>
                            </button>
                            {isConfirmingDelete ? (
                              <div className="inline-flex items-center gap-1">
                                <button
                                  type="button"
                                  onClick={() => void handleDeleteRecord(rec.id)}
                                  className="min-h-[34px] rounded-md bg-red-700 px-2.5 py-1 text-xs font-semibold text-white hover:bg-red-800"
                                >
                                  Confirm Delete
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setConfirmDeleteId(null)}
                                  className="min-h-[34px] rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-slate-700"
                                >
                                  Cancel
                                </button>
                              </div>
                            ) : (
                              <button
                                type="button"
                                onClick={() => setConfirmDeleteId(rec.id)}
                                className="inline-flex min-h-[34px] items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-50"
                              >
                                <Trash2 className="h-3 w-3" aria-hidden="true" />
                                <span>Delete</span>
                              </button>
                            )}
                          </div>
                        </div>
                        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
                          {rec.content}
                        </p>
                      </>
                    )}
                  </article>
                );
              })
            )}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Permanent Identity Anchors */}
          <div className="lg:col-span-5 rounded-xl border border-slate-200 bg-white p-6 space-y-5">
            <div>
              <h2 className="font-display text-base font-bold text-slate-900">
                Permanent Identity Foundations
              </h2>
              <p className="mt-1 text-xs text-slate-600">
                Immutable across conversations and application restarts.
              </p>
            </div>

            <dl className="divide-y divide-slate-200 text-sm">
              <div className="py-3 flex justify-between gap-4">
                <dt className="font-semibold text-slate-700">System Name</dt>
                <dd className="font-mono text-slate-900">{identity?.name ?? 'STAN AI'}</dd>
              </div>
              <div className="py-3 flex justify-between gap-4">
                <dt className="font-semibold text-slate-700">Owner / Creator</dt>
                <dd className="text-slate-900">{identity?.owner_name ?? 'Stan'}</dd>
              </div>
              <div className="py-3 flex justify-between gap-4">
                <dt className="font-semibold text-slate-700">Established Answer</dt>
                <dd className="text-right text-slate-900 font-medium">
                  You are Stan, Kalda&apos;s son.
                </dd>
              </div>
              <div className="py-3 flex justify-between gap-4">
                <dt className="font-semibold text-slate-700">Project</dt>
                <dd className="text-slate-900">{identity?.project_name ?? 'Kalda'}</dd>
              </div>
            </dl>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed text-slate-700">
              <p className="font-semibold text-slate-900">Transparency Commitment</p>
              <p className="mt-1">
                {identity?.system_nature ??
                  'STAN AI is a non-conscious software system combining an independent PyTorch decoder-only Transformer with persistent memory and explicit reasoning.'}
              </p>
            </div>
          </div>

          {/* Owner-Configurable Personality & Judgment */}
          <form
            onSubmit={handleSaveIdentity}
            className="lg:col-span-7 rounded-xl border border-slate-200 bg-white p-6 space-y-6"
          >
            <div>
              <h2 className="font-display text-base font-bold text-slate-900">
                Owner-Configurable Personality &amp; Autonomy
              </h2>
              <p className="mt-1 text-xs text-slate-600">
                Tune how STAN AI balances directness, skepticism, humor, warmth, and honest disagreement.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
              {(
                [
                  ['directness', 'Directness', 'Plain, unhedged communication'],
                  ['independence', 'Independence', 'Forms own reasoned stance'],
                  ['skepticism', 'Skepticism', 'Questions unverified assumptions'],
                  ['humor', 'Dry Humor', 'Contextual wit without forcing jokes'],
                  ['warmth', 'Warmth', 'Supportive tone when appropriate'],
                  ['verbosity', 'Verbosity', 'Concise vs. detailed explanations'],
                ] as const
              ).map(([key, label, hint]) => (
                <div key={key}>
                  <div className="flex items-center justify-between text-xs font-semibold text-slate-800">
                    <label htmlFor={`trait-${key}`}>{label}</label>
                    <span className="font-mono tabular-nums">{personality[key].toFixed(2)}</span>
                  </div>
                  <input
                    id={`trait-${key}`}
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={personality[key]}
                    onChange={(e) =>
                      setPersonality((prev) => ({ ...prev, [key]: Number(e.target.value) }))
                    }
                    className="mt-2 w-full accent-blue-700"
                  />
                  <p className="mt-1 text-[11px] text-slate-500">{hint}</p>
                </div>
              ))}
            </div>

            <div className="border-t border-slate-200 pt-4 space-y-2.5">
              <p className="text-xs font-semibold text-slate-900">Judgment &amp; Disagreement Rules</p>
              <label className="flex items-center gap-2.5 text-sm text-slate-800">
                <input
                  type="checkbox"
                  checked={personality.allow_disagreement}
                  onChange={(e) =>
                    setPersonality((prev) => ({ ...prev, allow_disagreement: e.target.checked }))
                  }
                  className="h-4 w-4 accent-blue-700"
                />
                <span>Allow honest disagreement (&ldquo;I disagree.&rdquo;, &ldquo;That does not make sense.&rdquo;)</span>
              </label>
              <label className="flex items-center gap-2.5 text-sm text-slate-800">
                <input
                  type="checkbox"
                  checked={personality.allow_bluntness}
                  onChange={(e) =>
                    setPersonality((prev) => ({ ...prev, allow_bluntness: e.target.checked }))
                  }
                  className="h-4 w-4 accent-blue-700"
                />
                <span>Allow blunt pushback against manipulation or disrespect</span>
              </label>
              <label className="flex items-center gap-2.5 text-sm text-slate-800">
                <input
                  type="checkbox"
                  checked={personality.challenge_bad_assumptions}
                  onChange={(e) =>
                    setPersonality((prev) => ({
                      ...prev,
                      challenge_bad_assumptions: e.target.checked,
                    }))
                  }
                  className="h-4 w-4 accent-blue-700"
                />
                <span>Challenge contradictory or unsupported assumptions</span>
              </label>
            </div>

            <div className="border-t border-slate-200 pt-4">
              <label htmlFor="core-principles-input" className="block text-xs font-semibold text-slate-900">
                Core Principles (one per line)
              </label>
              <textarea
                id="core-principles-input"
                rows={4}
                value={principlesText}
                onChange={(e) => setPrinciplesText(e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-xs leading-relaxed text-slate-900"
              />
            </div>

            {identityError && (
              <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs font-medium text-red-800">
                {identityError}
              </div>
            )}
            {identityStatus && (
              <div role="status" className="rounded-lg border border-emerald-200 bg-emerald-50 px-3.5 py-2.5 text-xs font-medium text-emerald-900">
                {identityStatus}
              </div>
            )}

            <div className="flex items-center justify-end gap-3">
              <button
                type="submit"
                disabled={identitySaving}
                className="min-h-[40px] rounded-lg bg-blue-700 px-5 py-2 text-xs font-semibold text-white hover:bg-blue-800 disabled:opacity-60"
              >
                {identitySaving ? 'Saving...' : 'Save Identity & Personality'}
              </button>
            </div>
          </form>
        </div>
      )}
    </section>
  );
}
